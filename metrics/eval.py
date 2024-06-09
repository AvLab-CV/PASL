"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
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

import math
import network

@torch.no_grad()
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



        print('Generating images ...')
        for i, x_src in enumerate(tqdm(loader_eval, total=len(loader_eval))):
            lm = x_src[4]
            lm = lm.to(device)
            x2_target_lm = x_src[3]
            x2_target = x_src[1]
            x2_target_lm = x2_target_lm.to(device)
            x2_target = x2_target.to(device)
            # gt = x_src[5]
            # gt = gt.to(device)
            N = x2_target_lm.size(0) #batch-size
            if args.masks:
                masks = x2_target_lm
            else:
                masks = None

            for j in range(args.num_outs_per_domain):

                x1_source = x_src[0]
                x1_source = x1_source.to(device)

                s_trg = nets.style_encoder(x1_source)

                x_fake = nets.generator(x2_target_lm, lm, s_trg, masks=masks)

                # save generated images to calculate FID later
                from torch.nn.functional import cosine_similarity
                #------------------------------------------------------------
                # csim = 0
                # for k in range(16):
                #     f1 = x_fake[k].flatten()
                #     pp = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                #     # pp = [0,1,2,3]
                #     for i in pp:
                #         if i != k:
                #             f2 = x_fake[i].flatten()
                #             # f2 /= f2.norm(dim=-1, keepdim=True)
                #             # f1 /= f1.norm(dim=-1, keepdim=True)
                #             f2 = f2.view(1,32768)
                #             f1 = f1.view(1,32768)
                #             similarity = cosine_similarity(f1, f2)
                #             csim += similarity.item()
                #             print(similarity.item())
                # print('CSIM: ',csim/240)
                # print('qq')
                # ------------------------------------------------------------------

                # save generated images to calculate FID later
                for k in range(N):
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

                    utils_lm.save_image(x_fake[k], ncol=1, filename=filename)
                    utils_lm.save_image(x1_source[k], ncol=1, filename=filename2)
                    utils_lm.save_image(x2_target_lm[k], ncol=1, filename=filename3)
                    utils_lm.save_image(x2_target[k], ncol=1, filename=filename4)

    #     calculate_fid_for_all_tasks(args, step=step, mode=mode)
    #     calculate_csim_for_all_tasks(args, step=step, mode=mode)
    #     calculate_ssim_for_all_tasks(args, step=step, mode=mode)
    #     #calculate_isim_for_all_tasks(args, step=step, mode=mode)
    # else :
    #     calculate_fid_for_all_tasks(args, step=step, mode=mode)
    #     calculate_csim_for_all_tasks(args, step=step, mode=mode)
    #     calculate_ssim_for_all_tasks(args, step=step, mode=mode)
    #     #calculate_isim_for_all_tasks(args, step=step, mode=mode)

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
#         x2_target = nn.functional.interpolate(x2_target[:, :, :, :], size=(112, 112), mode='bilinear')
#         criterion_id.eval()
#         with torch.torch.no_grad():
#             source_embs = criterion_id(x1_source)
#             target_embs = criterion_id(x2_target)
#             fake_embs = criterion_id(x_fake)

#         cos = nn.CosineSimilarity(dim=1, eps=1e-6)

#         output_1 = cos(source_embs, fake_embs)
#         csim_1 += torch.mean(output_1)

#         output_2 = cos(target_embs, fake_embs)
#         csim_2 += torch.mean(output_2)

#         output_3 = cos(source_embs, target_embs)
#         csim_3 += torch.mean(output_3)


#     csim_1 = csim_1 / len(loader)
#     csim_2 = csim_2 / len(loader)
#     csim_3 = csim_3 / len(loader)

#     csim_1_values['CSIM_source_fake_%s/%s' % (mode, iter)] = csim_1.item()
#     csim_2_values['CSIM_target_fake_%s/%s' % (mode, iter)] = csim_2.item()
#     csim_3_values['CSIM_target_source_%s/%s' % (mode, iter)] = csim_3.item()
#     filename = os.path.join(args.eval_dir, 'CSIM_source_fake_%.5i_%s.json' % (step, mode))
#     filename2 = os.path.join(args.eval_dir, 'CSIM_target_fake_%.5i_%s.json' % (step, mode))
#     filename3 = os.path.join(args.eval_dir, 'CSIM_target_source_%.5i_%s.json' % (step, mode))
#     utils_lm.save_json(csim_1_values, filename)
#     utils_lm.save_json(csim_2_values, filename2)
#     utils_lm.save_json(csim_3_values, filename3)





def calculate_csim_for_all_tasks(args, step, mode):

    iter = '%s' % (step)
    path_real = os.path.join(args.eval_dir, iter + 'ground_truth')
    path_fake = os.path.join(args.eval_dir, iter)
    path_source = os.path.join(args.eval_dir, iter + 'real')


    paths = [path_real, path_fake, path_source]

    img_size = args.img_size
    batch_size = args.val_batch_size
    print('Calculating CSIM given paths %s and %s...' % (paths[0], paths[1]))
    loader = get_eval_loader_2(paths, img_size, batch_size, imagenet_normalize=False)


    csim_1_values = OrderedDict()
    csim_2_values = OrderedDict()
    csim_3_values = OrderedDict()
    csim_1 = 0
    csim_2 = 0
    csim_3 = 0
    print("Loading Arcface model.....")
    BACKBONE_RESUME_ROOT = 'FR_Pretrained_Test/Pretrained/ms1m_ir50/backbone_ir50_ms1m_epoch120.pth'

    INPUT_SIZE = [112, 112]
    arcface = IR_50(INPUT_SIZE)

    if os.path.isfile(BACKBONE_RESUME_ROOT):
        arcface.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
        print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion_id = arcface.to(device)


    for x in tqdm(loader, total=len(loader)):


        x2_target = x[0]
        x_fake = x[1]
        x1_source = x[2]
        x2_target = x2_target.to(device)
        x_fake = x_fake.to(device)
        x1_source = x1_source.to(device)
        # gt = gt.to(device)

        x1_source = nn.functional.interpolate(x1_source[:, :, :, :], size=(112, 112), mode='bilinear')
        x_fake = nn.functional.interpolate(x_fake[:, :, :, :], size=(112, 112), mode='bilinear')
        x2_target = nn.functional.interpolate(x2_target[:, :, :, :], size=(112, 112), mode='bilinear')
        criterion_id.eval()
        with torch.torch.no_grad():
            source_embs = criterion_id(x1_source)
            target_embs = criterion_id(x2_target)
            fake_embs = criterion_id(x_fake)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        output_1 = cos(source_embs, fake_embs)
        csim_1 += torch.mean(output_1)

        output_2 = cos(target_embs, fake_embs)
        csim_2 += torch.mean(output_2)

        output_3 = cos(source_embs, target_embs)
        csim_3 += torch.mean(output_3)


    csim_1 = csim_1 / len(loader)
    csim_2 = csim_2 / len(loader)
    csim_3 = csim_3 / len(loader)

    csim_1_values['CSIM_source_fake_%s/%s' % (mode, iter)] = csim_1.item()
    csim_2_values['CSIM_target_fake_%s/%s' % (mode, iter)] = csim_2.item()
    csim_3_values['CSIM_target_source_%s/%s' % (mode, iter)] = csim_3.item()
    filename = os.path.join(args.eval_dir, 'CSIM_source_fake_%.5i_%s.json' % (step, mode))
    filename2 = os.path.join(args.eval_dir, 'CSIM_target_fake_%.5i_%s.json' % (step, mode))
    filename3 = os.path.join(args.eval_dir, 'CSIM_target_source_%.5i_%s.json' % (step, mode))
    utils_lm.save_json(csim_1_values, filename)
    utils_lm.save_json(csim_2_values, filename2)
    utils_lm.save_json(csim_3_values, filename3)





def calculate_fid_for_all_tasks(args, step, mode):
    print('Calculating FID for all tasks...')
    fid_values = OrderedDict()
    iter = '%s' % (step)
    # path_real = os.path.join(args.eval_dir, iter + 'ground_truth')
    path_source = os.path.join(args.eval_dir, iter + 'real')
    path_fake = os.path.join(args.eval_dir, iter)
    print('Calculating FID for %s...' % iter)
    fid_value = calculate_fid_given_paths(
        paths=[path_source, path_fake],
        img_size=args.img_size,
        batch_size=args.val_batch_size)
    fid_values['FID_%s/%s' % (mode, iter)] = fid_value

    # calculate the average FID for all tasks
    fid_mean = 0
    for _, value in fid_values.items():
        fid_mean += value / len(fid_values)
    fid_values['FID_%s/mean' % mode] = fid_mean

    # report FID values
    filename = os.path.join(args.eval_dir, 'FID_%.5i_%s.json' % (step, mode))
    utils_lm.save_json(fid_values, filename)

def calculate_ssim_for_all_tasks(args, step, mode):

    iter = '%s' % (step)
    path_real = os.path.join(args.eval_dir, iter + 'ground_truth')
    path_fake = os.path.join(args.eval_dir, iter)
    path_source = os.path.join(args.eval_dir, iter + 'real')


    paths = [path_real, path_fake, path_source]

    img_size = args.img_size
    batch_size = args.val_batch_size
    print('Calculating SSIM given paths %s and %s...' % (paths[0], paths[1]))
    loader = get_eval_loader_2(paths, img_size, batch_size, imagenet_normalize=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ssim_2_values = OrderedDict()
    ssim_2 = 0
    metric = SSIM()


    for x in tqdm(loader, total=len(loader)):

        x2_target = x[0]
        x_fake = x[1]

        x2_target = x2_target.to(device)
        x_fake = x_fake.to(device)

        ssim_2 += metric(x_fake, x2_target).item()




    ssim_2 = ssim_2 / len(loader)
    ssim_2_values['SSIM_target_fake_%s/%s' % (mode, iter)] = ssim_2
    filename2 = os.path.join(args.eval_dir, 'SSIM_target_fake_%.5i_%s.json' % (step, mode))
    utils_lm.save_json(ssim_2_values, filename2)





class SSIM(object):
    '''
    modified from https://github.com/jorge-pessoa/pytorch-msssim
    '''

    def __init__(self, des="structural similarity index"):
        self.des = des

    def __repr__(self):
        return "SSIM"

    def gaussian(self, w_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - w_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(w_size)])
        return gauss / gauss.sum()

    def create_window(self, w_size, channel=1):
        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window

    def __call__(self, y_pred, y_true, w_size=11, size_average=True, full=False):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            w_size : int, default 11
            size_average : boolean, default True
            full : boolean, default False
        return ssim, larger the better
        """
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if torch.max(y_pred) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(y_pred) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val

        padd = 0
        (_, channel, height, width) = y_pred.size()
        window = self.create_window(w_size, channel=channel).to(y_pred.device)

        mu1 = F.conv2d(y_pred, window, padding=padd, groups=channel)
        mu2 = F.conv2d(y_true, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(y_pred * y_pred, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(y_true * y_true, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(y_pred * y_true, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret
