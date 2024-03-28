"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from copy import deepcopy
import math
from typing import Optional
from munch import Munch
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn as nn
import torch.nn.functional as F

# visT
import logging
import torch.nn as nn
from fastai.vision import *

# from .ABINet_main.modules.attention import *
# from .ABINet_main.modules.backbone import ResTranformer
# from .ABINet_main.modules.model import Model
# from .ABINet_main.modules.resnet import resnet45

# from core.wing import FAN
import torchvision.transforms as transforms
# from core.resnet50_ft_dims_2048 import resnet50_ft
from torch.cuda.amp import autocast


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        # self.fc1 = nn.Linear(64*512, style_dim) #modified deleted

        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        # h = self.fc1(s) #modified deleted

        h = self.fc(s) #modified s->h
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class NoiseLayer(nn.Module):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight"""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None

    def forward(self, x, noise=None):

        if noise is None and self.noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        elif noise is None:
            # print(x.device)
            # here is a little trick: if you get all the noise layers and set each
            # modules .noise attribute, you can have pre-defined noise.
            # Very useful for analysis
            noise = self.noise
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x
        
class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, self_att=False, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        self.self_att = self_att
        repeat_num = int(np.log2(img_size)) - 2
        if self.self_att:
            for _ in range(repeat_num-2):
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out
            blocks += [SelfAttention(dim_out)]
            for _ in range(2): #####Ori 2
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out
        else:
            for _ in range(repeat_num):
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]

        self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        self.shared = nn.Sequential(*blocks)

        self.fc = nn.Linear(dim_out, style_dim)


    def forward(self, x):

        h = self.shared(x)
        h = h.view(h.size(0), -1)
        h = self.fc(h)

        return h





class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)
        self.dim_in = dim_in
        self.dim_out = dim_out



    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)

        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):

        # print(x)
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)

        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1],
                                    [-1, 8., -1],
                                    [-1, -1, -1]]).to(device) / w_hpf

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))

# region General Blocks

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.rand(1).normal_(0.0, 0.02))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        # B: mini batches, C: channels, W: width, H: height
        B, C, H, W = x.shape
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(B, -1, W * H)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(B, -1, W * H)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        out = self.gamma * out + x

        return out
# Transformer        
TN_HIDDEN_DIM=512
TN_DROPOUT = 0.1
TN_NHEADS = 8
TN_DIM_FEEDFORWARD = 512
TN_ENC_LAYERS = 3
TN_DEC_LAYERS = 3
class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1):
        super().__init__()
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(6, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(TN_HIDDEN_DIM, TN_NHEADS, TN_DIM_FEEDFORWARD,
                                                TN_DROPOUT, "relu", True)
        encoder_norm = nn.LayerNorm(TN_HIDDEN_DIM) if True else None
        self.trands_encoder = TransformerEncoder(encoder_layer, TN_ENC_LAYERS, encoder_norm)

        #Transformer decoder

        decoder_layer = TransformerDecoderLayer(TN_HIDDEN_DIM, TN_NHEADS, TN_DIM_FEEDFORWARD,
                                                TN_DROPOUT, "relu", True)
        decoder_norm = nn.LayerNorm(TN_HIDDEN_DIM)
        
        # self.before_query = nn.Linear(8*8*512, 16*512) #for self-att
        # self.query_embed =nn.Linear(16*512, 512) #for cross-att
        self.query_embed =nn.Linear(128, 512) #for temp
        # self.query_embed =nn.Linear(8*8*512, 512)

        self.trans_decoder = TransformerDecoder(decoder_layer, TN_DEC_LAYERS, decoder_norm,
                                          return_intermediate=True)
        self.linear_q = nn.Linear(TN_DIM_FEEDFORWARD, TN_DIM_FEEDFORWARD*8)
        self.conv = torch.nn.Conv2d(512,512,(1,1),1,(2,3))
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 3, 1, 1, 0))
        self.to_rgb_lm = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 4, 1, 1, 0))

        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        #print('repeat_num', repeat_num)
        if w_hpf > 0:
            repeat_num += 1

        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            # dim_out = 8192 #temp
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(3):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    def forward(self, x, lm, s, masks=None):
        x = torch.cat((x, lm), dim=1)
        x = self.from_rgb(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)
        output_x=x
        # print(x.shape)
        # self_x = x.view(2,8,8,512)  #for self attention 
        FEAT_ST_ENC = x.flatten(2).permute(2,0,1)
        
        
        memory = self.trands_encoder(FEAT_ST_ENC)
        
        # s=s.view(4,128) #for cross att (for adain)
        # s =s.view(128) # temp

        # self_s = self_x.view(2,8*8*512) #for self att 
        # self_s = self.before_query(self_s) #for self att
        # print(s)
        QR_EMB = self.query_embed(s).unsqueeze(0) #for cross-att
        # QR_EMB = self.query_embed(self_s).unsqueeze(0) #for self-att

        tgt = torch.zeros_like(QR_EMB)  
        hs_sk = self.trans_decoder(tgt, memory, query_pos = QR_EMB)  

        hs_sk = hs_sk.transpose(1, 2)[-1] #torch.cat([hs.transpose(1, 2)[-1], QR_EMB.permute(1,0,2)], -1)

        hs_sk = self.linear_q(hs_sk)
        hs_sk = hs_sk.contiguous()
      
        hs_sk = hs_sk.view(hs_sk.size(0), 2, 4, -1)
        hs_sk = hs_sk.permute(0, 3, 2, 1)
        x=self.conv(hs_sk)
        # for block in self.decode:
        #     print()
        #     print(block)
        for block in self.decode:
            x = block(x, s) #for cross-att
            # x = block(x, self_x) #for self-att

            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[:, 0:1, :, :]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])

        output =torch.sigmoid(self.to_rgb(x))

        return output
class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, y_ind):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)

        y_emb = query_embed[y_ind].permute(1,0,2)

        tgt = torch.zeros_like(y_emb)
        memory = self.encoder(src)
        hs = self.decoder(tgt, memory, query_pos=y_emb)
                        
        return torch.cat([hs.transpose(1, 2)[-1], y_emb.permute(1,0,2)], -1)

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask= None,
                src_key_padding_mask = None,
                pos= None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask = None,
                memory_mask= None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,
                pos = None,
                query_pos= None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask= None,
                     src_key_padding_mask= None,
                     pos= None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask = None,
                    src_key_padding_mask= None,
                    pos = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask= None,
                src_key_padding_mask = None,
                pos= None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        value=tgt2
        tgt2 = self.self_attn(q, k, value, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)




def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class Discriminator_img_pix(nn.Module):
    def __init__(self, img_size=256, num_domains=2, self_att=False, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        self.self_att = self_att
        self.input = nn.Conv2d(9, dim_in, 3, 1, 1)
        self.blocks = nn.ModuleList()


        repeat_num = int(np.log2(img_size)) - 2
        if self.self_att:
            for _ in range(repeat_num-2):
                dim_out = min(dim_in*2, max_conv_dim)
                self.blocks.append(ResBlk(dim_in, dim_out, downsample=True))
                dim_in = dim_out
            self.blocks.append(SelfAttention(dim_out))
            for _ in range(2):
                dim_out = min(dim_in*2, max_conv_dim)
                self.blocks.append(ResBlk(dim_in, dim_out, downsample=True))
                dim_in = dim_out
        else:
            for _ in range(repeat_num):
                dim_out = min(dim_in*2, max_conv_dim)
                self.blocks.append(ResBlk(dim_in, dim_out, downsample=True))
                dim_in = dim_out

        self.output = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(dim_out, dim_out, 4, 1, 0), nn.LeakyReLU(0.2),  nn.Conv2d(dim_out, 1, 1, 1, 0))


    def forward(self, x, y,z):
        out = torch.cat((x, y,z), dim=1)
        out = self.input(out)
        fea = []
        for num,block in enumerate(self.blocks):
            out = block(out)
            fea.append(out)
        out =  self.output(out)
        out = out.view(out.size(0), -1)  # (batch, num_domains)

        return fea, out

        
# class BaseVision(Model):
#     def __init__(self):
#         super().__init__(config=None)
      
#         self.backbone = resnet45()
#         self.attention = PositionAttention()


#     def forward(self, images, *args):
#         #----------------------------------------------------------------------------------
#         VGG = resnet50_ft(weights_path='./FR_Pretrained_Test/Pretrained/VGGFace2/resnet50_ft_dims_2048.pth') #modified
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         facemodel = VGG.to(device)

#         images = images.to(device)
#         with torch.no_grad():
#             features = facemodel(nn.functional.interpolate(images, size=(224, 224), mode='bilinear')*255)
#         # 將特徵表示轉換為 numpy 陣列
#         features = features[0]
#         features_array = features.squeeze()
#         # print(features_array.shape)
#         #-------------------------------------------------------------------------------------

#         features = self.backbone(images)  # (N, E, H, W)
#         attn_vecs, attn_scores = self.attention(features, features_array)  # (N, T, E), (N, T, H, W)

#         return { attn_vecs,attn_scores}


class Discriminator_img2_pix(nn.Module):
    def __init__(self, img_size=256, num_domains=2, self_att=False, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        self.self_att = self_att
        self.input = nn.Conv2d(9, dim_in, 3, 1, 1)
        self.blocks = nn.ModuleList()

        repeat_num = int(np.log2(img_size)) - 2
        if self.self_att:
            for _ in range(repeat_num-2):
                dim_out = min(dim_in*2, max_conv_dim)
                self.blocks.append(ResBlk(dim_in, dim_out, downsample=True))
                dim_in = dim_out
            self.blocks.append(SelfAttention(dim_out))
            for _ in range(2):
                dim_out = min(dim_in*2, max_conv_dim)
                self.blocks.append(ResBlk(dim_in, dim_out, downsample=True))
                dim_in = dim_out
        else:
            for _ in range(repeat_num):
                dim_out = min(dim_in*2, max_conv_dim)
                self.blocks.append(ResBlk(dim_in, dim_out, downsample=True))
                dim_in = dim_out

        self.output = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(dim_out, dim_out, 2, 1, 0), nn.LeakyReLU(0.2),  nn.Conv2d(dim_out, 1, 1, 1, 0))


    def forward(self, x, y,z):
        x = F.avg_pool2d(x, 2)
        y = F.avg_pool2d(y, 2)
        z = F.avg_pool2d(z, 2)
        out = torch.cat((x, y,z), dim=1)
        out = self.input(out)
        fea = []
        for num,block in enumerate(self.blocks):

            out = block(out)
            fea.append(out)
        out =  self.output(out)
        out = out.view(out.size(0), -1)  # (batch, num_domains)

        return fea, out

class Discriminator_img(nn.Module):
    def __init__(self, img_size=256, num_domains=2, self_att=False, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(7, dim_in, 3, 1, 1)]
        self.self_att = self_att
        repeat_num = int(np.log2(img_size)) - 2
        if self.self_att:
            for _ in range(repeat_num-2):
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out
            blocks += [SelfAttention(dim_out)]
            for _ in range(2):
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out
        else:
            for _ in range(repeat_num):
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, 1, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)


    def forward(self, x, y, z):
        out = torch.cat((x, y, z), dim=1)
        out = self.main(out)
        out = out.view(out.size(0), -1)  # (batch, num_domains)

        return out

######################################
class Discriminator_img_lm(nn.Module):
    def __init__(self, img_size=256, num_domains=2, self_att=False, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(6, dim_in, 3, 1, 1)]
        self.self_att = self_att
        repeat_num = int(np.log2(img_size)) - 2
        if self.self_att:
            for _ in range(repeat_num-2):
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out
            blocks += [SelfAttention(dim_out)]
            for _ in range(2):
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out
        else:
            for _ in range(repeat_num):
                dim_out = min(dim_in*2, max_conv_dim)
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
                dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, 1, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)


    def forward(self, x, y):


        x = x.to(device)
        out = torch.cat((x, y), dim=1)
        out = self.main(out)
        out = out.view(out.size(0), -1)  # (batch, num_domains)

        return out




def build_model(args):

    if args.multi_discriminator:
        generator = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf)
        # style_encoder = BaseVision() # ViT- style
        style_encoder = StyleEncoder(args.img_size, args.style_dim, args.num_domains, args.self_att)
        discriminator_img = Discriminator_img_pix(args.img_size, args.num_domains, args.self_att)
        discriminator_img2 = Discriminator_img2_pix(args.img_size, args.num_domains, args.self_att)

        generator_ema = deepcopy(generator)
        style_encoder_ema = deepcopy(style_encoder)


        nets = Munch(generator=generator,
                     style_encoder=style_encoder,
                     discriminator=discriminator_img,discriminator2=discriminator_img2)
        nets_ema = Munch(generator=generator_ema,
                         style_encoder=style_encoder_ema)
    # if args.landmark_loss:
    #     generator = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf)
    #     style_encoder = StyleEncoder(args.img_size, args.style_dim, args.num_domains, args.self_att)
    #     discriminator_img = Discriminator_img(args.img_size, args.num_domains, args.self_att)
    #     discriminator_img_lm = Discriminator_img_lm(args.img_size, args.num_domains, args.self_att)
    #     generator_ema = copy.deepcopy(generator)
    #     style_encoder_ema = copy.deepcopy(style_encoder)


    #     nets = Munch(generator=generator,
    #                  style_encoder=style_encoder,
    #                  discriminator=discriminator_img, discriminator2=discriminator_img_lm)
    #     nets_ema = Munch(generator=generator_ema,
    #                      style_encoder=style_encoder_ema)
    
    else:
        generator = Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf)
        style_encoder = StyleEncoder(args.img_size, args.style_dim, args.num_domains, args.self_att)
        discriminator_img = Discriminator_img(args.img_size, args.num_domains, args.self_att)
        generator_ema = copy.deepcopy(generator)
        style_encoder_ema = copy.deepcopy(style_encoder)


        nets = Munch(generator=generator,
                     style_encoder=style_encoder,
                     discriminator=discriminator_img)
        nets_ema = Munch(generator=generator_ema,
                         style_encoder=style_encoder_ema)

    # if args.w_hpf > 0:
    #     fan = FAN(fname_pretrained=args.wing_path).eval()
    #     nets.fan = fan
    #     nets_ema.fan = fan

    return nets, nets_ema
