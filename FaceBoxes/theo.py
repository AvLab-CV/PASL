# SOTAs, single template
import argparse
import numpy as np
import os
import time
from skimage import transform as trans
import cv2
import math
from util.reference_pts_SOTA import Reference_Points
from util.Load_Lmrk_Pose_Img import Load_Lmrk_Pose, Load_Img
import sys
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
#from utils.render_ctypes import render  # faster
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix
import yaml
import os
from sklearn.metrics.pairwise import cosine_similarity
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '4'
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
cfg = yaml.load(open('/media/avlab/viva/configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
tddfa = TDDFA(gpu_mode='gpu', **cfg)
face_boxes = FaceBoxes()

pose={}
# (yaw angle <= 30) = src_temp_img / (yaw angle > 30) = ref_temp_img
src_temp_img = []
src_temp_angle = []
ref_temp_img=[] 
ref_temp_angle = []
# find ref with diff id each src
src_temp2_img = []
src_temp2_angle = []
ref_temp2_img=[] 
ref_temp2_angle = []
# final list for src & ref & gt
src_img = []
src_angle = []
ref_img=[] 
ref_angle = []
gt_img=[] 
gt_angle = []
# for final storing
tmp_file_src = {'img_path':[], 'yaw_angle':[]}
tmp_file_ref = {'img_path':[], 'yaw_angle':[]}
tmp_file_gt = {'img_path':[], 'yaw_angle':[]}
tran_list = open('/media/avlab/7974ac65-96ab-45fd-8d7c-1156efbc7d66/Database/vox2_all_face_crop/vox2_large_pose_data.txt', 'w')

for i in range(8):
    i += 1
    i = i * 1000000
    Pose_List = np.load('./_Pose_List/{}_Pose_{}.npy'.format('Voxel1', i), allow_pickle=True).item()
    pose.update(Pose_List)
print(pose[0])
count = 0

'''seperate original data with <=30 and >40'''
for idx, file_idx in enumerate(pose): 
    if idx % 3 == 0: #decrease pose
        yaw_angle = pose[file_idx][0]*180/math.pi
        if (abs(yaw_angle) <=30):
            # print(file_idx)
            src_temp_img.append(file_idx)
            src_temp_angle.append(yaw_angle)
            # count += 1
        elif (abs(yaw_angle) > 40): 
            ref_temp_img.append(file_idx)
            ref_temp_angle.append(yaw_angle)
            # count += 1
        else:continue
    else:continue

# for i in range(1000000): #decrease src data
#     del pose[-1]5

print(len(ref_temp_angle))
print(len(src_temp_angle))
print('====================================')


'''find ref'''
for idy, file_idy in enumerate(src_temp_img): #(all 590w)(current 28.1w)
    n1 = 0
    # print(file_idy.split('/'))
    for idz in range(len(ref_temp_img)): #(all 100w)(current 14.4w)
        if (idz+2)%(idy+3)==0:
            if ref_temp_img[idz].split('/')[1] != file_idy.split('/')[1]: # find 7 ref for each src diff id
                # print(ref_temp_img[idz].split('/')[1])
                n1 += 1
                ref_temp2_img.append(ref_temp_img[idz])
                ref_temp2_angle.append(ref_temp_angle[idz])
                src_temp2_img.append(file_idy)
                src_temp2_angle.append(src_temp_angle[idy])
                if n1 == 10: break #7
        else: continue

print('src temp2')
print(len(src_temp2_img))
print(len(src_temp2_angle))
print('ref temp2')
print(len(ref_temp2_img))
print(len(ref_temp2_angle))

tran_list.write('#start')
tran_list.write('\n')

'''find gt'''
for i in range(len(ref_temp2_img)): #(all 4000w)(current 140w)
    # path_ref = '/media/avlab/viva/database/'+ ref_temp2_img[i] + '.jpg' #this is the output ref
    # print(ref_temp2_img[i])
    path_ref = '/media/avlab/7974ac65-96ab-45fd-8d7c-1156efbc7d66/Database/vox2_all_face_crop/'+ ref_temp2_img[i] + '.jpg' #this is the output ref
    path_ref = path_ref.replace("Vox2_5fps","crop_256")
    # print(path_ref)
    para_ref = cv2.imread(path_ref)
    if para_ref is None:
        continue
    boxes_ref = face_boxes(para_ref)
    z = len(boxes_ref)
    if z == 0:
        print("no face error")
        # sys.exit(-1)
        continue
    # print(f'detect {z} faces for all ref')
    # print(i)
    ref_param_lst, roi_box_lst1 = tddfa(para_ref, boxes_ref)
    ref_camera, ref_exp =ref_param_lst[0][0:12], ref_param_lst[0][52:63]

    flag1 = 0 #how many gt per ref
    for idk,item in enumerate(ref_temp_img): # (all 100w)(current 14.4w) go to original data with yaw>40(which is ref_temp_img)to search for gt
        if (item.split('/')[2] == src_temp2_img[i].split('/')[2]): #same video with scr
            # print('===================================')
            # print(item.split('/')[2])
            # print(src_temp2_img[i].split('/')[2])
            # print('===================================')
            # path_gt = '/media/avlab/viva/database/'+ item +'.jpg'
            path_gt = '/media/avlab/7974ac65-96ab-45fd-8d7c-1156efbc7d66/Database/vox2_all_face_crop/'+ item +'.jpg'
            path_gt = path_gt.replace("Vox2_5fps","crop_256")
            para_gt=cv2.imread(path_gt)
            if para_gt is None:
                continue
            boxes_gt = face_boxes(para_gt)
            n = len(boxes_gt)
            if n == 0:
                print("no face error")
                # sys.exit(-1)
                continue
            gt_param_lst, roi_box_lst = tddfa(para_gt, boxes_gt)
            gt_camera, gt_exp =gt_param_lst[0][0:12], gt_param_lst[0][52:63]
            cosine_sim_camera=cosine_similarity([gt_camera],[ref_camera])
            cosine_sim_exp=cosine_similarity([gt_exp],[ref_exp])

            if cosine_sim_camera>=0.99 and cosine_sim_exp>=0.99:
                gt_img.append(item)
                gt_angle.append(ref_temp_angle[idk])
                src_img.append(src_temp2_img[i])
                src_angle.append(src_temp2_angle[i])
                ref_img.append(ref_temp2_img[i])
                ref_angle.append(ref_temp2_angle[i])
                flag1 += 1
                if flag1 > 3: break
                # print(f'detect {flag1} faces for gt')

            else: continue
        else: continue

print('\n')
print('GT')
print(len(gt_img))
print(len(gt_angle))
print('\n')
print('ref')
print(len(ref_img))
print(len(ref_angle))
print('\n')
print('src')
print(len(src_img))
print(len(src_angle))
print('\n')


for s in range(len(gt_img)):
    first_img = src_img[s]
    second_img = gt_img[s]
    third_img = ref_img[s]
    tran_list.write(first_img + '.jpg' + ' ' + second_img + '.jpg'+' '+ third_img+'.jpg')
    tran_list.write('\n')



print('Completed')
tran_list.close()