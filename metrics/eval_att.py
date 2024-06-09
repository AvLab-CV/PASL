"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import shutil

import torchvision.utils as vutils
                                  
from fastai.vision import *
from torch.nn.parallel import DistributedDataParallel
import torchvision.transforms as transforms
from .ABINet_main.utils import CharsetMapper,blend_mask


import os
from re import T
import shutil
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.fid import calculate_fid_given_paths
from metrics.lpips import calculate_lpips_given_images
from core.data_loader_lm_perceptual import get_eval_loader_vgg, get_eval_loader_2
from core import utils_lm
import cv2
from ms1m_ir50.model_irse import IR_50
import math
import network

from PIL import Image, ImageDraw

def calculate_metrics(nets, args, step, mode):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 

    #read the testing image
    loader_eval = get_eval_loader_vgg(root=args.val_img_dir,
                                      train_data=args.dataset,
                                      img_size=args.img_size,
                                      batch_size=args.val_batch_size,
                                      imagenet_normalize=False,
                                      drop_last=True, mode = args.mode)
    iter = '%s' % (step)
    if not os.path.exists(os.path.join(args.eval_dir, iter)):

        path_fake = os.path.join(args.eval_dir, iter)
        shutil.rmtree(path_fake, ignore_errors=True)
        os.makedirs(path_fake)

        path_real = os.path.join(args.eval_dir, iter + 'real')
        shutil.rmtree(path_real, ignore_errors=True)
        os.makedirs(path_real)

        path_real_lm = os.path.join(args.eval_dir, iter + 'lm')
        shutil.rmtree(path_real_lm, ignore_errors=True)
        os.makedirs(path_real_lm)

        path_ground_truth_lm = os.path.join(args.eval_dir, iter + 'ground_truth')
        shutil.rmtree(path_ground_truth_lm, ignore_errors=True)
        os.makedirs(path_ground_truth_lm)

        path_fake1= os.path.join(args.eval_dir, iter+ 'att')
        shutil.rmtree(path_fake1, ignore_errors=True)
        os.makedirs(path_fake1)

     
        print('Generating images ...')
        for i, x_src in enumerate(tqdm(loader_eval, total=len(loader_eval))):
            lm=x_src[4]
            lm=lm.to(device)
            x2_target_lm = x_src[3]
            x2_target = x_src[1]
            x2_target_lm = x2_target_lm.to(device)
            x2_target = x2_target.to(device)
            N = x2_target_lm.size(0) #batch-size
            
            if args.masks:
                masks = x2_target_lm
            else:
                masks = None

            for j in range(args.num_outs_per_domain):

                pil = transforms.ToPILImage()
                tensor = transforms.ToTensor()
                size = 256,256
                resize = transforms.Resize(size=size, interpolation=0)
                
                x1_source = x_src[0]
                x1_source = x1_source.to(device)

                s_trg,att = nets.style_encoder(x1_source)
                s_trg1,att1 = nets.style_encoder(lm)
                x_fake = nets.generator(x2_target_lm,lm, s_trg, masks=masks)
                
                for z in att1:
                    for k in range(2):
                        image_np = np.array(pil(x1_source[k]))
                        attn_pil = [pil(a) for a in z[:, None, :, :]]
                        att= [tensor(resize(a)).to(device).repeat(3, 1, 1) for a in attn_pil]
                        attn_sum = np.array([np.array(a) for a in attn_pil[3:12]]).sum(axis=0)
                        
                        blended_sum =tensor(blend_mask(image_np, attn_sum)).to(device)
                        blended = [tensor(blend_mask(image_np, np.array(a))).to(device) for a in attn_pil]
                        x_fake1 = torch.stack([x1_source[k]] + att+ [blended_sum] + blended)
                        x_fake1=x_fake1.view(2, -1, *x_fake1.shape[1:])
                        x_fake1= x_fake1.permute(1, 0, 2, 3, 4).flatten(0, 1)
                # save generated images to calculate FID later
                        
                        filename = os.path.join(
                            path_fake,
                            '%.4i_%.2i.png' % (i * args.val_batch_size + (k + 1), j + 1))
                        filename2 = os.path.join(
                            path_real,
                            '%.4i_%.2i.png' % (i * args.val_batch_size + (k + 1), j + 1))
                        filename3 = os.path.join(
                            path_real_lm,
                            '%.4i_%.2i.png' % (i * args.val_batch_size + (k + 1), j + 1))
                        filename4 = os.path.join(
                            path_ground_truth_lm,
                            '%.4i_%.2i.png' % (i * args.val_batch_size + (k + 1), j + 1))
                        
                        filename5 = os.path.join(
                            path_fake1,
                            '%.4i_%.2i.png' % (i * args.val_batch_size + (k + 1), j + 1))
                        utils_lm.save_image(x_fake[k], ncol=1, filename=filename)
                        utils_lm.save_image(x1_source[k], ncol=1, filename=filename2)
                        utils_lm.save_image(lm[k], ncol=1, filename=filename3)
                        utils_lm.save_image(x2_target[k], ncol=1, filename=filename4)
                        utils_lm.save_image(x_fake1, ncol=1, filename=filename5)
                            

        calculate_fid_for_all_tasks(args, step=step, mode=mode)
        calculate_csim_for_all_tasks(args, step=step, mode=mode)
        calculate_ssim_for_all_tasks(args, step=step, mode=mode)
        #calculate_isim_for_all_tasks(args, step=step, mode=mode)
    else :
        calculate_fid_for_all_tasks(args, step=step, mode=mode)
        calculate_csim_for_all_tasks(args, step=step, mode=mode)
        calculate_ssim_for_all_tasks(args, step=step, mode=mode)
    #calculate_isim_for_all_tasks(args, step=step, mode=mode)




    

# def calculate_isim_for_all_tasks(args, step, mode):

#     iter = '%s' % (step)
#     path_real = os.path.join(args.eval_dir, iter + 'ground_truth')
#     path_fake = os.path.join(args.eval_dir, iter)
#     path_source = os.path.join(args.eval_dir, iter + 'real')


#     paths = [path_real, path_fake, path_source]

#     img_size = args.img_size
#     batch_size = args.val_batch_size
#     print('Calculating CSIM given paths %s and %s...' % (paths[0], paths[1]))
#     loader = get_eval_loader_2(paths, img_size, batch_size, imagenet_normalize=False)


#     csim_1_values = OrderedDict()
#     csim_2_values = OrderedDict()
#     csim_3_values = OrderedDict()
#     csim_1 = 0
#     csim_2 = 0
#     csim_3 = 0
#     print("Loading VGG model.....")
#     BACKBONE = resnet50_ft(weights_path='./FR_Pretrained_Test/Pretrained/VGGFace2/resnet50_ft_dims_2048.pth')

#     INPUT_SIZE = [112, 112]
#     arcface = IR_50(INPUT_SIZE)

#     if os.path.isfile(BACKBONE_RESUME_ROOT):
#         arcface.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
#         print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     criterion_id = arcface.to(device)


#     for x in tqdm(loader, total=len(loader)):


#         x2_target = x[0]
#         x_fake = x[1]
#         x1_source = x[2]
#         x2_target = x2_target.to(device)
#         x_fake = x_fake.to(device)
#         x1_source = x1_source.to(device)


#         x1_source = nn.functional.interpolate(x1_source[:, :, :, :], size=(112, 112), mode='bilinear')
#         x_fake = nn.functional.interpolate(x_fake[:, :, :, :], size=(112, 112), mode='bilinear')
#         x2_target = nn.functio