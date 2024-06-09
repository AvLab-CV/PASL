import os
import shutil
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2

# from metrics.fid import calculate_fid_given_paths
# from metrics.lpips import calculate_lpips_given_images
# from core.data_loader_lm_perceptual import get_eval_loader_vgg, get_eval_loader_2
# from core import utils_lm
from PIL import Image
from ms1m_ir50.model_irse import IR_50
import math
# import network


def calculate_csim_for_all_tasks(fake_folder, gt_folder, real_folder):


    convert_tensor = transforms.ToTensor()
    csim_values = OrderedDict()
    csim_= 0
    print("Loading Arcface model.....")
    BACKBONE_RESUME_ROOT = './ms1m_ir50/backbone_ir50_ms1m_epoch63.pth'

    INPUT_SIZE = [112, 112]
    arcface = IR_50(INPUT_SIZE)

    if os.path.isfile(BACKBONE_RESUME_ROOT):
        arcface.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
        print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion_id = arcface.to(device)
    fake_gt_csim_list = []
    fake_real_csim_list = []
    real_gt_csim_list = []
    fake_files = os.listdir(fake_folder)
    gt_files = os.listdir(gt_folder)
    real_files = os.listdir(real_folder)

    # 遍历两个文件夹中的图像并计算旋转误差
    for fake_file, gt_file, real_file in tqdm(zip(fake_files, gt_files, real_files), desc="Processing"):
        fake_image_path = os.path.join(fake_folder, fake_file)
        gt_image_path = os.path.join(gt_folder, gt_file)
        real_image_path = os.path.join(real_folder, real_file)
        if (fake_file == gt_file):

            try:
                fake_img = cv2.imread(fake_image_path)
                fake_img  = cv2.resize(fake_img, (112, 112))
                gt_img = cv2.imread(gt_image_path)
                gt_img  = cv2.resize(gt_img, (112, 112))
                real_img = cv2.imread(real_image_path)
                real_img  = cv2.resize(real_img, (112, 112))
                fake_img = convert_tensor(fake_img).to(device).reshape(1,3,112,112)
                gt_img = convert_tensor(gt_img).to(device).reshape(1,3,112,112)
                real_img = convert_tensor(real_img).to(device).reshape(1,3,112,112)


                fake_img = nn.functional.interpolate(fake_img[:, :, :], size=(112, 112), mode='bilinear')
                gt_img = nn.functional.interpolate(gt_img[:, :, :], size=(112, 112), mode='bilinear')
                real_img = nn.functional.interpolate(real_img[:, :, :], size=(112, 112), mode='bilinear')
                criterion_id.eval()
                with torch.torch.no_grad():
                    fake_embs = criterion_id(fake_img)
                    gt_embs = criterion_id(gt_img)
                    real_embs = criterion_id(real_img)

                cos = nn.CosineSimilarity(dim=1, eps=1e-6)

                fake_gt_csim = cos(fake_embs, gt_embs)
                fake_gt_csim_list.append(fake_gt_csim)
                fake_real_csim = cos(fake_embs, real_embs)
                fake_real_csim_list.append(fake_real_csim)
                real_gt_csim = cos(real_embs, gt_embs)
                real_gt_csim_list.append(real_gt_csim)
            except:
                print('No face detected error')
    print('fake_gt_csim:',sum(fake_gt_csim_list)/len(fake_gt_csim_list))
    print('fake_real_csim:',sum(fake_real_csim_list)/len(fake_real_csim_list))
    print('real_gt_csim:',sum(real_gt_csim_list)/len(real_gt_csim_list))


fake_image_folder = '/media/avlab/DATA/RFG_DECA_TRANS/expr/eval/cvpr_vox2/206000'
gt_image_folder = '/media/avlab/DATA/RFG_DECA_TRANS/expr/eval/cvpr_vox2/206000ground_truth'
real_image_folder = '/media/avlab/DATA/RFG_DECA_TRANS/expr/eval/cvpr_vox2/206000real'

calculate_csim_for_all_tasks(fake_image_folder, gt_image_folder, real_image_folder)