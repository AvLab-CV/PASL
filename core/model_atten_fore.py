# author: Niwhskal
# github : https://github.com/Niwhskal/SRNet

import os
# from skimage.io import imread, imsave
# from skimage.transform import resize
# import skimage
import numpy as np
import torch
import torch.nn as nn
from models_transformer.transformer import *
from models_transformer.blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock
import torchvision.transforms as tf
from torchvision.models import vgg19, resnet18
from collections import namedtuple
from data_loader import LABEL
import cfg
from modeling_segformer import SegformerModel,SegformerDecodeHead


temp_shape = (0,0)

class Self_Attn(torch.nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = torch.nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = torch.nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = torch.nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))

        self.softmax  = torch.nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out, attention

def calc_padding(h, w, k, s):
    
    h_pad = (((h-1)*s) + k - h)//2 
    w_pad = (((w-1)*s) + k - w)//2
    
    return (h_pad, w_pad)

def calc_inv_padding(h, w, k, s):
    h_pad = (k-h + ((h-1)*s))//2
    w_pad = (k-w + ((w-1)*s))//2
    
    return (h_pad, w_pad)


class Conv_bn_block(torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self._conv = torch.nn.Conv2d(*args, **kwargs)
        self._bn = torch.nn.BatchNorm2d(kwargs['out_channels'])
        
    def forward(self, input):
        return torch.nn.functional.leaky_relu(self._bn(self._conv(input)),negative_slope=0.2)

class Res_block(torch.nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
            
        self._conv1 = torch.nn.Conv2d(in_channels, in_channels//4, kernel_size = 1, stride =1)
        self._conv2 = torch.nn.Conv2d(in_channels//4, in_channels//4, kernel_size = 3, stride = 1, padding = 1)
        
        self._conv3 = torch.nn.Conv2d(in_channels//4, in_channels, kernel_size = 1, stride=1)
        
        self._bn = torch.nn.BatchNorm2d(in_channels)
       
    def forward(self, x):
        
        xin = x
        x = torch.nn.functional.leaky_relu(self._conv1(x),negative_slope=0.2)
        x = torch.nn.functional.leaky_relu(self._conv2(x),negative_slope=0.2)
        x = self._conv3(x)
        x = torch.add(xin, x)
        x = torch.nn.functional.leaky_relu(self._bn(x),negative_slope=0.2)
        
        return x

class encoder_net(torch.nn.Module):
    
    def __init__(self, in_channels, get_feature_map = False):
        super().__init__()
            
        self.cnum = 32
        self.get_feature_map = get_feature_map
        self._conv1_1 = Conv_bn_block(in_channels = in_channels, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self._conv1_2 = Conv_bn_block(in_channels =self.cnum, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        #--------------------------
        
        self._pool1 = torch.nn.Conv2d(in_channels = self.cnum, out_channels = 2 * self.cnum, kernel_size = 3, stride = 2, padding = calc_padding(temp_shape[0], temp_shape[1], 3, 2))
        
        self._conv2_1 = Conv_bn_block(in_channels = 2* self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self._conv2_2 = Conv_bn_block(in_channels = 2* self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        #---------------------------
        
        self._pool2 = torch.nn.Conv2d(in_channels = 2*self.cnum, out_channels = 4* self.cnum, kernel_size = 3, stride = 2, padding = calc_padding(temp_shape[0], temp_shape[1], 3, 2))
        
        self._conv3_1 = Conv_bn_block(in_channels = 4* self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self._conv3_2 = Conv_bn_block(in_channels = 4* self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        #---------------------------
        
        self._pool3 = torch.nn.Conv2d(in_channels = 4*self.cnum, out_channels = 8* self.cnum, kernel_size = 3, stride = 2, padding = calc_padding(temp_shape[0], temp_shape[1], 3, 2))
        
        self._conv4_1 = Conv_bn_block(in_channels = 8* self.cnum, out_channels = 8*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self._conv4_2 = Conv_bn_block(in_channels = 8* self.cnum, out_channels = 8*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
    def forward(self, x):
                
        x = self._conv1_1(x)
        x = self._conv1_2(x)
        
        x = torch.nn.functional.leaky_relu(self._pool1(x),negative_slope=0.2)
        x = self._conv2_1(x)
        x = self._conv2_2(x)
        
        f1 = x
        
        x = torch.nn.functional.leaky_relu(self._pool2(x),negative_slope=0.2)
        x = self._conv3_1(x)
        x = self._conv3_2(x)
        
        f2 = x
        
        x = torch.nn.functional.leaky_relu(self._pool3(x),negative_slope=0.2)
        x = self._conv4_1(x)
        x = self._conv4_2(x)
        
        
        if self.get_feature_map:
            return x, [f2, f1]
        
        else:
            return x
        
        
class build_res_block(torch.nn.Module):
    
    def __init__(self, in_channels):
        
        super().__init__()
        
        self._block1 = Res_block(in_channels)
        self._block2 = Res_block(in_channels)
        self._block3 = Res_block(in_channels)
        self._block4 = Res_block(in_channels)
        
    def forward(self, x):
        
        x = self._block1(x)
        x = self._block2(x)
        x = self._block3(x)
        x = self._block4(x)
        
        return x
    
    
class decoder_net(torch.nn.Module):
    
    def __init__(self, in_channels, get_feature_map = False, mt =1, fn_mt=1):
        super().__init__()
        
        self.cnum = 32
       
        self.get_feature_map = get_feature_map
        
        self._conv1_1 = Conv_bn_block(in_channels = fn_mt*in_channels , out_channels = 8*self.cnum, kernel_size = 3, stride =1, padding = 1) 

        self._conv1_2 = Conv_bn_block(in_channels = 8*self.cnum, out_channels = 8*self.cnum, kernel_size = 3, stride =1, padding = 1)
        
        #-----------------
        
        self._deconv1 = torch.nn.ConvTranspose2d(8*self.cnum, 4*self.cnum, kernel_size = 3, stride = 2, padding = calc_inv_padding(temp_shape[0], temp_shape[1], 3, 2))
        
        self._conv2_1 = Conv_bn_block(in_channels = fn_mt*mt*4*self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self._conv2_2 = Conv_bn_block(in_channels = 4*self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        #-----------------
        
        self._deconv2 = torch.nn.ConvTranspose2d(4*self.cnum, 2*self.cnum, kernel_size =3 , stride = 2, padding = calc_inv_padding(temp_shape[0], temp_shape[1], 3, 2))
        
        self._conv3_1 = Conv_bn_block(in_channels = fn_mt*mt*2*self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self._conv3_2 = Conv_bn_block(in_channels = 2*self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
       
        #----------------
        
        self._deconv3 = torch.nn.ConvTranspose2d(2*self.cnum, self.cnum, kernel_size =3 , stride = 2, padding = calc_inv_padding(temp_shape[0], temp_shape[1], 3, 2))
        
        self._conv4_1 = Conv_bn_block(in_channels = self.cnum, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)
        self._conv4_2 = Conv_bn_block(in_channels = self.cnum, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)

        self.attn1 = Self_Attn( 4*self.cnum, 'relu')
        self.attn2 = Self_Attn( 2*self.cnum,  'relu')
        
        
    def forward(self, x, fuse = None):
        
        
        if fuse and fuse[0] is not None:
            x = torch.cat((x, fuse[0]), dim = 1)
            
        x = self._conv1_1(x)
        x = self._conv1_2(x)
        f1 = x
        
        #----------
        
        x = torch.nn.functional.leaky_relu(self._deconv1(x), negative_slope=0.2)
       
        if fuse and fuse[1] is not None:
            x = torch.cat((x, fuse[1]), dim = 1)
            
        x = self._conv2_1(x)
        x = self._conv2_2(x)
        x,p1 = self.attn1(x)
        f2 = x
        
        #----------
        
        x = torch.nn.functional.leaky_relu(self._deconv2(x), negative_slope=0.2)
        if fuse and fuse[2] is not None:
            x = torch.cat((x, fuse[2]), dim = 1)
        
        x = self._conv3_1(x)
        x = self._conv3_2(x)
        x,p2 = self.attn2(x)
        f3 = x
        
        #----------
        
        x = torch.nn.functional.leaky_relu(self._deconv3(x), negative_slope=0.2)
        x = self._conv4_1(x)
        x = self._conv4_2(x)
        
        if self.get_feature_map:
            return x, [f1, f2, f3]        
        
        else:
            return x
                                                  
        
class Input_style_net(torch.nn.Module):
    
    def __init__(self, in_channels, get_feature_map = False, stn_use = True):
        super().__init__()
        
        self.get_feature_map = get_feature_map
        self.stn_use = stn_use
        
        self.cnum = 32
        self._t_encoder = encoder_net(in_channels)
        self._t_res = build_res_block(8*self.cnum)
        
        self._s_encoder = encoder_net(3)
        self._s_res = build_res_block(8*self.cnum)
        
        self._sk_decoder = decoder_net(16*self.cnum)
        self._sk_out = torch.nn.Conv2d(self.cnum, 1, kernel_size =3, stride = 1, padding = 1)
        
        self._t_decoder = decoder_net(16*self.cnum)
        self._t_cbr = Conv_bn_block(in_channels = self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self._t_out = torch.nn.Conv2d(self.cnum, 3, kernel_size = 3, stride = 1, padding = 1)

        self.attn1 = Self_Attn( 512, 'relu')

        
    def forward(self, x_t, x_s):

        x_s = self._s_encoder(x_s)
        x_s = self._s_res(x_s)

        x_t = self._t_encoder(x_t)
        x_t = self._t_res(x_t)

        x = torch.cat((x_t, x_s), dim = 1)
        x,p1 = self.attn1(x)

        # y_sk = self._sk_decoder(x, fuse = None)
        # y_sk_out = torch.sigmoid(self._sk_out(y_sk))        
        
        y_t = self._t_decoder(x, fuse = None)
        
        # y_t = torch.cat((y_sk, y_t), dim = 1)
        # y_t = self._t_cbr(y_t)
        y_t_out = torch.tanh(self._t_out(y_t))

        # if   self.get_feature_map == True:
        #     return y_sk_out, y_t_out, x_t
        # else:
        #     return y_sk_out, y_t_out
        if   self.get_feature_map == True:
            return  y_t_out, x_t
        else:
            return  y_t_out
                                          
        
class inpainting_net(torch.nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.cnum = 32
        self._encoder = encoder_net(4, get_feature_map = True)
        self._res = build_res_block(8*self.cnum)
        
        self._decoder = decoder_net(8*self.cnum,  get_feature_map = True, mt=2)
        self._out = torch.nn.Conv2d(self.cnum, 3, kernel_size = 3, stride = 1, padding = 1)
        
    def forward(self, x):
        
        x, f_encoder = self._encoder(x)
        x = self._res(x)

        x, fs = self._decoder(x, fuse = [None] + f_encoder)
        
        x = torch.tanh(self._out(x))
        
        return x, fs
  
        
class fusion_net(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cnum = 32
        
        self._encoder = encoder_net(in_channels)
        self._res = build_res_block(8*self.cnum)
        
        self._decoder = decoder_net(8*self.cnum, fn_mt = 2)
        
        self._out = torch.nn.Conv2d(self.cnum, 3, kernel_size = 3, stride = 1, padding = 1)
         
    def forward(self, x, fuse):
        
        x = self._encoder(x)
        x = self._res(x)
        x = self._decoder(x, fuse = fuse)
        x = torch.tanh(self._out(x))
        
        return x
           
# class Generator(torch.nn.Module):
    
#     def __init__(self, in_channels):
        
#         super().__init__()
        
#         self.cnum = 32
        
#         self._tcn = text_conversion_net(in_channels)
        
#     def forward(self, i_t, i_sm, gbl_shape):
                
#         temp_shape = gbl_shape

#         o_sk, o_t = self._tcn(i_t, i_sm)
        
        
#         return o_sk, o_t
    
    
class Discriminator(torch.nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.cnum = 32
        self._conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size = 3, stride = 2, padding = calc_padding(temp_shape[0], temp_shape[1], 3, 2))
        self._conv2 = torch.nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = calc_padding(temp_shape[0], temp_shape[1], 3, 2))
    
        self._conv2_bn = torch.nn.BatchNorm2d(128)
        
        self._conv3 = torch.nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = calc_padding(temp_shape[0], temp_shape[1], 3, 2))
        
        self._conv3_bn = torch.nn.BatchNorm2d(256)
        
        self._conv4 = torch.nn.Conv2d(256, 512, kernel_size = 3, stride = 2, padding = calc_padding(temp_shape[0], temp_shape[1], 3, 2))
        
        self._conv4_bn = torch.nn.BatchNorm2d(512)
        
        self._conv5 = torch.nn.Conv2d(512, 1,  kernel_size = 3, stride = 1, padding = 1)
        
        self._conv5_bn = torch.nn.BatchNorm2d(1)

        self.attn1 = Self_Attn( 256, 'relu')
        self.attn2 = Self_Attn( 512,  'relu')
        
     
    def forward(self, x):
        
        x = torch.nn.functional.leaky_relu(self._conv1(x), negative_slope=0.2)
        x = self._conv2(x)
        x = torch.nn.functional.leaky_relu(self._conv2_bn(x), negative_slope=0.2)
        x = self._conv3(x)
        x = torch.nn.functional.leaky_relu(self._conv3_bn(x), negative_slope=0.2)
        x,p1 = self.attn1(x)
        x = self._conv4(x)
        
        x = torch.nn.functional.leaky_relu(self._conv4_bn(x), negative_slope=0.2)
        x,p2 = self.attn2(x)
        x = self._conv5(x)
        x = self._conv5_bn(x)
        
        return x
        
    
class Vgg19(torch.nn.Module):
    def __init__(self):
        
        super(Vgg19, self).__init__()
        features = list(vgg19(pretrained = True).features)
        self.features = torch.nn.ModuleList(features).eval()
        
    def forward(self, x):
        
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            
            if ii in {1, 6, 11, 20, 29}:
                results.append(x)
        return results

class Conv_block(torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self._conv = torch.nn.Conv2d(*args, **kwargs)
        
    def forward(self, input):
        return torch.nn.functional.leaky_relu(self._conv(input),negative_slope=0.2)

class decoder_resnet_net(torch.nn.Module):
    
    def __init__(self,):
        super().__init__()

        self.cnum = 16

        self._deconv1 = torch.nn.ConvTranspose2d(2048, 2048, kernel_size = 3, stride = 2, padding = 1, output_padding=1)
        self._conv1_1 = Conv_block(in_channels = 2048 , out_channels = 1024, kernel_size = 3, stride =1, padding = 1) 

        self._deconv2 = torch.nn.ConvTranspose2d(1024, 1024, kernel_size = 3, stride = 2, padding = 1, output_padding=1)
        self._conv2_1 = Conv_block(in_channels = 1024 , out_channels = 512, kernel_size = 3, stride =1, padding = 1) 

        self._deconv3 = torch.nn.ConvTranspose2d(512, 512, kernel_size = 3, stride = 2, padding = 1, output_padding=1)
        self._conv3_1 = Conv_block(in_channels = 512 , out_channels = 256, kernel_size = 3, stride =1, padding = 1) 

        self._deconv4 = torch.nn.ConvTranspose2d(256, 256, kernel_size = 3, stride = 2, padding = 1, output_padding=1)
        self._conv4_1 = Conv_block(in_channels = 256 , out_channels = 64, kernel_size = 3, stride =1, padding = 1) 

        self._deconv5 = torch.nn.ConvTranspose2d(64, 64, kernel_size = 3, stride = 2, padding = 1, output_padding=1)
        self._conv5_1 = Conv_block(in_channels = 64 , out_channels = 64, kernel_size = 3, stride =1, padding = 1) 

       
    def forward(self, x):
        
        y = torch.nn.functional.leaky_relu(self._deconv1(x[4]), negative_slope=0.2)
        y = self._conv1_1(y)
        y = torch.add(y,x[3])

        y = torch.nn.functional.leaky_relu(self._deconv2(y), negative_slope=0.2)
        y = self._conv2_1(y)
        y = torch.add(y,x[2])

        y = torch.nn.functional.leaky_relu(self._deconv3(y), negative_slope=0.2)
        y = self._conv3_1(y)
        y = torch.add(y,x[1])

        y = torch.nn.functional.leaky_relu(self._deconv4(y), negative_slope=0.2)
        y = self._conv4_1(y)
        y = torch.add(y,x[0])

        y = torch.nn.functional.leaky_relu(self._deconv5(y), negative_slope=0.2)
        y = self._conv5_1(y)
        
        return y

class Bottleneck(nn.Module):
    """
    __init__
        in_channel：残差块输入通道数
        out_channel：残差块输出通道数
        stride：卷积步长
        downsample：在_make_layer函数中赋值，用于控制shortcut图片下采样 H/2 W/2
    """
    expansion = 4   # 残差块第3个卷积层的通道膨胀倍率
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)   # H,W不变。C: in_channel -> out_channel
        self.bn1 = nn.BatchNorm2d(num_features=out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, bias=False, padding=1)  # H/2，W/2。C不变
        self.bn2 = nn.BatchNorm2d(num_features=out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)   # H,W不变。C: out_channel -> 4*out_channel
        self.bn3 = nn.BatchNorm2d(num_features=out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        identity = x    # 将原始输入暂存为shortcut的输出
        if self.downsample is not None:
            identity = self.downsample(x)   # 如果需要下采样，那么shortcut后:H/2，W/2。C: out_channel -> 4*out_channel(见ResNet中的downsample实现)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity     # 残差连接
        out = self.relu(out)

        return out

class ResNet50(nn.Module):
    """
    __init__
        block: 堆叠的基本模块
        block_num: 基本模块堆叠个数,是一个list,对于resnet50=[3,4,6,3]
        num_classes: 全连接之后的分类特征维度
        
    _make_layer
        block: 堆叠的基本模块
        channel: 每个stage中堆叠模块的第一个卷积的卷积核个数，对resnet50分别是:64,128,256,512
        block_num: 当期stage堆叠block个数
        stride: 默认卷积步长
    """
    def __init__(self, block, block_num, num_classes=1000):
        super(ResNet50, self).__init__()
        self.in_channel = 64    # conv1的输出维度

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)     # H/2,W/2。C:3->64
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)     # H/2,W/2。C不变
        self.layer1 = self._make_layer(block=block, channel=64, block_num=block_num[0], stride=1)   # H,W不变。downsample控制的shortcut，out_channel=64x4=256
        self.layer2 = self._make_layer(block=block, channel=128, block_num=block_num[1], stride=2)  # H/2, W/2。downsample控制的shortcut，out_channel=128x4=512
        self.layer3 = self._make_layer(block=block, channel=256, block_num=block_num[2], stride=2)  # H/2, W/2。downsample控制的shortcut，out_channel=256x4=1024
        self.layer4 = self._make_layer(block=block, channel=512, block_num=block_num[3], stride=2)  # H/2, W/2。downsample控制的shortcut，out_channel=512x4=2048

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))  # 将每张特征图大小->(1,1)，则经过池化后的输出维度=通道数
        self.fc = nn.Linear(in_features=512*block.expansion, out_features=num_classes)

        for m in self.modules():    # 权重初始化
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None   # 用于控制shorcut路的
        if stride != 1 or self.in_channel != channel*block.expansion:   # 对resnet50：conv2中特征图尺寸H,W不需要下采样/2，但是通道数x4，因此shortcut通道数也需要x4。对其余conv3,4,5，既要特征图尺寸H,W/2，又要shortcut维度x4
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=channel*block.expansion, kernel_size=1, stride=stride, bias=False), # out_channels决定输出通道数x4，stride决定特征图尺寸H,W/2
                nn.BatchNorm2d(num_features=channel*block.expansion))

        layers = []  # 每一个convi_x的结构保存在一个layers列表中，i={2,3,4,5}
        layers.append(block(in_channel=self.in_channel, out_channel=channel, downsample=downsample, stride=stride)) # 定义convi_x中的第一个残差块，只有第一个需要设置downsample和stride
        self.in_channel = channel*block.expansion   # 在下一次调用_make_layer函数的时候，self.in_channel已经x4

        for _ in range(1, block_num):  # 通过循环堆叠其余残差块(堆叠了剩余的block_num-1个)
            layers.append(block(in_channel=self.in_channel, out_channel=channel))

        return nn.Sequential(*layers)   # '*'的作用是将list转换为非关键字参数传入

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return [x0,x1,x2,x3,x4]

class Input_conversion_net(torch.nn.Module):
    
    def __init__(self,in_channels):
        super().__init__()
        
        self.cnum = 32

        # self._encoder_sk = effnetv2_m(in_channels=3)
        # self._decoder_sk = decoder_effient_net()
        # self._mask_out = torch.nn.Conv2d(24, 1, kernel_size =3, stride = 1, padding = 1)


        self._encoder_sk = ResNet50(block=Bottleneck, block_num=[3, 4, 6, 3])
        self._decoder_sk = decoder_resnet_net()
        self._mask_out = torch.nn.Conv2d(64, 1, kernel_size =3, stride = 1, padding = 1)
        
        # self._s_encoder = encoder_net(in_channels)
        # self._s_res = build_res_block(8*self.cnum)
        
        # self._sk_decoder = decoder_net(8*self.cnum)
        # self._sk_out = torch.nn.Conv2d(self.cnum, 1, kernel_size =3, stride = 1, padding = 1)
        
    def forward(self, x_s, gbl_shape):
                
        x_sk = self._encoder_sk(x_s)
        y_sk = self._decoder_sk(x_sk)
        y_sk_out = torch.sigmoid(self._mask_out(y_sk))

        # x_s = self._s_encoder(x_s)
        # x_s = self._s_res(x_s)

        # y_sk = self._sk_decoder(x_s, fuse = None)
        # y_sk_out = torch.sigmoid(self._sk_out(y_sk))        
        
        
        return y_sk_out       

class TextEncoder_FC(torch.nn.Module):
    def __init__(self, text_max_len):
        super(TextEncoder_FC, self).__init__()
        embed_size = 64
        self.embed = nn.Embedding(103, embed_size)
        self.fc = nn.Sequential(
                nn.Linear(text_max_len*embed_size, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=False),
                nn.Linear(1024, 2048),
                nn.BatchNorm1d(2048),
                nn.ReLU(inplace=False),
                nn.Linear(2048, 4096)
                )
        '''embed content force'''
        self.linear = nn.Linear(embed_size, 256) 

    def forward(self, x, f_xs_shape):
        xx = self.embed(x) # b,t,embed
        batch_size = xx.shape[0]
        # xxx = xx.reshape(batch_size, -1) # b,t*embed
        # out = self.fc(xxx)

        '''embed content force'''
        xx_new = self.linear(xx) # b, text_max_len, 512
        ts = xx_new.shape[1]
        height_reps = f_xs_shape[-2]
        width_reps = f_xs_shape[-1] // ts
        if width_reps == 0:
            width_reps = 1
        
        tensor_list = list()
        for i in range(ts):
            text = [xx_new[:, i:i + 1]] # b, text_max_len, 512
            tmp = torch.cat(text * width_reps, dim=1)
            tensor_list.append(tmp)

        padding_reps = f_xs_shape[-1] % ts

        if padding_reps:
            embedded_padding_char = self.embed(torch.full((1, 1), 2, dtype=torch.long).cuda())
            embedded_padding_char = self.linear(embedded_padding_char)
            padding = embedded_padding_char.repeat(batch_size, padding_reps, 1)
            tensor_list.append(padding)

        res = torch.cat(tensor_list, dim=1) # b, text_max_len * width_reps + padding_reps, 512
        res = res.permute(0, 2, 1).unsqueeze(2) # b, 512, 1, text_max_len * width_reps + padding_reps
        final_res = torch.cat([res] * height_reps, dim=2)
        # print(f_xs_shape,final_res.shape)
        return final_res         

class Input_style_net_label(torch.nn.Module):
    
    def __init__(self, in_channels, get_feature_map = False, stn_use = True):
        super().__init__()
        
        self.get_feature_map = get_feature_map
        self.stn_use = stn_use
        
        self.cnum = 32
        self._t_encoder = TextEncoder_FC(12)
        self._t_res = build_res_block(8*self.cnum)
        
        self._s_encoder = encoder_net(3)
        self._s_res = build_res_block(8*self.cnum)
        
        self._sk_decoder = decoder_net(16*self.cnum)
        self._sk_out = torch.nn.Conv2d(self.cnum, 1, kernel_size =3, stride = 1, padding = 1)
        
        self._t_decoder = decoder_net(16*self.cnum)
        self._t_cbr = Conv_bn_block(in_channels = 2*self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self._t_out = torch.nn.Conv2d(self.cnum, 3, kernel_size = 3, stride = 1, padding = 1)

        self.attn1 = Self_Attn( 512, 'relu')

        
    def forward(self, x_t, x_s):

        x_s = self._s_encoder(x_s)
        x_s = self._s_res(x_s)

        x_t = self._t_encoder(x_t,x_s.shape)
        # x_t = self._t_res(x_t)

        x = torch.cat((x_t, x_s), dim = 1)
        x,p1 = self.attn1(x)

        # y_sk = self._sk_decoder(x, fuse = None)
        # y_sk_out = torch.sigmoid(self._sk_out(y_sk))        
        
        y_t = self._t_decoder(x, fuse = None)
        
        # y_t = torch.cat((y_sk, y_t), dim = 1)
        # y_t = self._t_cbr(y_t)
        y_t_out = torch.tanh(self._t_out(y_t))

        if   self.get_feature_map == True:
            return y_sk_out, y_t_out, x_t
        else:
            return y_t_out
TN_HIDDEN_DIM = 512
TN_DROPOUT = 0.1
TN_NHEADS = 8
TN_DIM_FEEDFORWARD = 512
TN_ENC_LAYERS = 3
TN_DEC_LAYERS = 3
# LABEL = ["\u00B7",'\u7696','\u6CAA','\u6D25','\u6E1D','\u5180','\u664B','\u8499','\u8FBD','\u5409',"\u9ED1","\u82CF","\u6D59",'\u4EAC','\u95FD','\u8D63','\u9C81', "\u8C6B","\u9102","\u6E58","\u6FB3","\u6842","\u743C",'\u5DDD','\u8D35','\u4E91','\u85CF','\u9655','\u7518','\u9752','\u5B81','\u65B0','\u8B66','\u5B66','\u0041','\u0042','\u0043','\u0044','\u0045','\u0046','\u0047','\u0048','\u004A','\u004B','\u004C','\u004D','\u004E','\u0050','\u0051','\u0052','\u0053','\u0054','\u0055','\u0056','\u0057','\u0058','\u0059','\u005A','\u0030', '\u0031','\u0032','\u0033','\u0034','\u0035','\u0036','\u0037','\u0038','\u0039','\u004F']
VOCAB_SIZE = len(LABEL) + 1
ALL_CHARS = False
ADD_NOISE = False

class FCNDecoder(nn.Module):
    def __init__(self, ups=4, n_res=2, dim=512, out_dim=1, res_norm='adain', activ='relu', pad_type='reflect'):
        super(FCNDecoder, self).__init__()
        self.conv = torch.nn.Conv2d(dim,dim,(5,7),1,(2,0))
        self.relu = torch.nn.ReLU(inplace=False)
        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm,
                                 activ, pad_type=pad_type)]
        for i in range(ups):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm='in',
                                       activation=activ,
                                       pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='sigmoid',
                                   pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x =  self.relu(self.conv(x))
        y =  self.model(x)
        

        return y

class TRGAN(nn.Module):

    def __init__(self,config):
        super(TRGAN, self).__init__()

        # encoder 

        # self.Feat_Encoder = nn.Sequential(*([nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)] +list(resnet18(pretrained=True).children())[1:-2]))
        
        # Segformer encoder 
        
        self.Feat_Encoder = SegformerModel(config)

        encoder_layer = TransformerEncoderLayer(TN_HIDDEN_DIM, TN_NHEADS, TN_DIM_FEEDFORWARD,
                                                TN_DROPOUT, "relu", True)
        encoder_norm = nn.LayerNorm(TN_HIDDEN_DIM) if True else None
        self.encoder = TransformerEncoder(encoder_layer, TN_ENC_LAYERS, encoder_norm)

        # sk decoder

        decoder_layer = TransformerDecoderLayer(TN_HIDDEN_DIM, TN_NHEADS, TN_DIM_FEEDFORWARD,
                                                TN_DROPOUT, "relu", True)
        decoder_norm = nn.LayerNorm(TN_HIDDEN_DIM)

        self.decoder = TransformerDecoder(decoder_layer, TN_DEC_LAYERS, decoder_norm,
                                          return_intermediate=True)
        self.linear_q = nn.Linear(TN_DIM_FEEDFORWARD, TN_DIM_FEEDFORWARD*8)

        self.query_embed = nn.Embedding(VOCAB_SIZE, TN_HIDDEN_DIM)
        
        self.DEC = FCNDecoder(res_norm = 'in')

        # o_m decoder
        
        self.decode_head = SegformerDecodeHead(config)

        

        self._muE = nn.Linear(512,512)
        self._logvarE = nn.Linear(512,512)         

        self._muD = nn.Linear(512,512)
        self._logvarD = nn.Linear(512,512)   


        self.l1loss = nn.L1Loss()

        self.noise = torch.distributions.Normal(loc=torch.tensor([0.]), scale=torch.tensor([1.0]))


        



    def reparameterize(self, mu, logvar):

        mu = torch.unbind(mu , 1)
        logvar = torch.unbind(logvar , 1)

        outs = []

        for m,l in zip(mu, logvar):
       
            sigma = torch.exp(l)
            eps = torch.cuda.FloatTensor(l.size()[0],1).normal_(0,1)
            eps  = eps.expand(sigma.size())

            out = m + sigma*eps

            outs.append(out)


        return torch.stack(outs, 1)


    # def Eval(self, ST, QRS):

    #     # if IS_SEQ:
    #     #     B, N, R, C = ST.shape
    #     #     FEAT_ST = self.Feat_Encoder(ST.view(B*N, 1, R, C))
    #     #     FEAT_ST = FEAT_ST.view(B, 512, 1, -1)
    #     # else:
    #     #     FEAT_ST = self.Feat_Encoder(ST)

    #     FEAT_ST = self.Feat_Encoder(ST)

    #     FEAT_ST_ENC = FEAT_ST.flatten(2).permute(2,0,1)

    #     memory = self.encoder(FEAT_ST_ENC)

    #     # if IS_KLD:

    #     #     Ex = memory.permute(1,0,2)

    #     #     memory_mu = self._muE(Ex)
    #     #     memory_logvar = self._logvarE(Ex)

    #     #     memory = self.reparameterize(memory_mu, memory_logvar).permute(1,0,2)

        
    #     OUT_IMGS = []

    #     for i in range(QRS.shape[1]):
            
    #         QR = QRS[:, i, :]

    #         if ALL_CHARS:    
    #             QR_EMB = self.query_embed.weight.repeat(batch_size,1,1).permute(1,0,2)
    #         else:
    #             QR_EMB = self.query_embed.weight[QR].permute(1,0,2)

    #         tgt = torch.zeros_like(QR_EMB)
            
    #         hs = self.decoder(tgt, memory, query_pos=QR_EMB)

    #         if IS_KLD:

    #             Dx = hs[0].permute(1,0,2)

    #             hs_mu = self._muD(Dx)
    #             hs_logvar = self._logvarD(Dx)

    #             hs = self.reparameterize(hs_mu, hs_logvar).permute(1,0,2).unsqueeze(0)

                            
    #         h = hs.transpose(1, 2)[-1]#torch.cat([hs.transpose(1, 2)[-1], QR_EMB.permute(1,0,2)], -1)
    #         if ADD_NOISE: h = h + self.noise.sample(h.size()).squeeze(-1).cuda()

    #         h = self.linear_q(h)
    #         h = h.contiguous()

    #         if ALL_CHARS: h = torch.stack([h[i][QR[i]] for i in range(batch_size)], 0)

    #         h = h.view(h.size(0), h.shape[1]*2, 4, -1)
    #         h = h.permute(0, 3, 2, 1)

    #         h = self.DEC(h)

          
    #         OUT_IMGS.append(h.detach())



    #     return OUT_IMGS
        


    


    def forward(self, QR, ST, QRs = None, mode = 'train'):

        #Attention Visualization Init    


        enc_attn_weights, dec_attn_weights = [], []

        self.hooks = [
         
            self.encoder.layers[-1].self_attn.register_forward_hook(
                lambda self, input, output: enc_attn_weights.append(output[1])
            ),
            self.decoder.layers[-1].multihead_attn.register_forward_hook(
                lambda self, input, output: dec_attn_weights.append(output[1])
            ),
        ]
        

        #Attention Visualization Init 

        # B, N, R, C = ST.shape
        # FEAT_ST = self.Feat_Encoder(ST)
        # FEAT_ST = FEAT_ST.view(B, 512, 1, -1)
        outputs = self.Feat_Encoder(
            ST,
            output_attentions=False,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=True,
        )

        encoder_hidden_states = outputs.hidden_states

        logits = self.decode_head(encoder_hidden_states)

        FEAT_ST = outputs[0]
        FEAT_ST_ENC = FEAT_ST.flatten(2).permute(2,0,1)

        memory = self.encoder(FEAT_ST_ENC)

        QR_EMB = self.query_embed.weight[QR].permute(1,0,2) 

        tgt = torch.zeros_like(QR_EMB)
        
        hs_sk = self.decoder(tgt, memory, query_pos = QR_EMB)                       

        hs_sk = hs_sk.transpose(1, 2)[-1] #torch.cat([hs.transpose(1, 2)[-1], QR_EMB.permute(1,0,2)], -1)

        if ADD_NOISE: h = h + self.noise.sample(h.size()).squeeze(-1).cuda()

        hs_sk = self.linear_q(hs_sk)
        hs_sk = hs_sk.contiguous()

        hs_sk = hs_sk.view(hs_sk.size(0), hs_sk.shape[1], 8, -1)
        hs_sk = hs_sk.permute(0, 3, 2, 1)

        hs_sk = self.DEC(hs_sk)

        
        self.dec_attn_weights = dec_attn_weights[-1].detach()
        self.enc_attn_weights = enc_attn_weights[-1].detach()


                
        for hook in self.hooks:
            hook.remove()

        return logits,hs_sk
    