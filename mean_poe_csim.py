import os
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
from ms1m_ir50.model_irse import IR_50
import math
from math import cos, sin, atan2, asin
import numpy as np
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from TDDFA_ONNX import TDDFA_ONNX
import yaml
# 加载3DDFA-V2模型配置
cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

# 初始化FaceBoxes和TDDFA
face_boxes = FaceBoxes_ONNX()
tddfa = TDDFA_ONNX(**cfg)


def P2sRt(P):
    """ decompositing camera matrix P.
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t2d: (2,). 2d translation.
    """
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d
def matrix2angle(R):
    """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    refined by: https://stackoverflow.com/questions/43364900/rotation-matrix-to-euler-angles-with-opencv
    todo: check and debug
     Args:
         R: (3,3). rotation matrix
     Returns:
         x: yaw
         y: pitch
         z: roll
     """
    if R[2, 0] > 0.998:
        z = 0
        x = np.pi / 2
        y = z + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        z = 0
        x = -np.pi / 2
        y = -z + atan2(R[0, 1], R[0, 2])
    else:
        x = asin(R[2, 0])
        y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

    return x, y, z
def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(radian). The same as in 3DDFA.
    Args:
        angles: [3,]. x, y, z angles
        x: yaw.
        y: pitch.
        z: roll.
    Returns:
        R: 3x3. rotation matrix.
    '''
    # x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x, y, z = angles[0], angles[1], angles[2]
    y, x, z = angles[0], angles[1], angles[2]

    # x
    Rx = np.array([[1, 0, 0],
                   [0, cos(x), -sin(x)],
                   [0, sin(x), cos(x)]])
    # y
    Ry = np.array([[cos(y), 0, sin(y)],
                   [0, 1, 0],
                   [-sin(y), 0, cos(y)]])
    # z
    Rz = np.array([[cos(z), -sin(z), 0],
                   [sin(z), cos(z), 0],
                   [0, 0, 1]])
    R = Rz.dot(Ry).dot(Rx)
    return R.astype(np.float32)





# 定义函数计算两张图像之间的旋转误差
def calculate_rotation(image):

    # 使用FaceBoxes检测人脸
    boxes = face_boxes(image)
    
    n = len(boxes)
    if n == 0:
        print(f'No face detected, exit')
        return None

    # 使用3DDFA-V2进行3D姿势估计
    param_lst, roi_box_lst = tddfa(image, boxes)
    
    try:
        param_lst, roi_box_lst = tddfa(image, boxes)
        param = param_lst[0]
    except Exception as e:
        print(f'3D pose estimation failed, skipping')
        return None
    
    P1 = param[:12].reshape(3, -1).copy()  # camera matrix
    s, R1, t3d = P2sRt(P1)
    angle = matrix2angle(R1)
    yaw, pitch, roll = angle
    return yaw* (180/math.pi)


def calculate_csim_for_all_tasks(fake_folder, gt_folder, real_folder):

    convert_tensor = transforms.ToTensor()
    csim_values = OrderedDict()
    csim_= 0
    print("Loading Arcface model.....")
    BACKBONE_RESUME_ROOT_FF = './POE/FF/Backbone_IR_50_Epoch_80.pth'

    INPUT_SIZE = [112, 112]
    arcface_ff = IR_50(INPUT_SIZE)

    if os.path.isfile(BACKBONE_RESUME_ROOT_FF):
        arcface_ff.load_state_dict(torch.load(BACKBONE_RESUME_ROOT_FF))
        print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT_FF))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion_id_ff = arcface_ff.to(device)


    BACKBONE_RESUME_ROOT_FS = './POE/FS/Backbone_IR_50_Epoch_80.pth'

    arcface_fs = IR_50(INPUT_SIZE)

    if os.path.isfile(BACKBONE_RESUME_ROOT_FS):
        arcface_fs.load_state_dict(torch.load(BACKBONE_RESUME_ROOT_FS))
        print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT_FS))

    criterion_id_fs = arcface_fs.to(device)


    BACKBONE_RESUME_ROOT_FP = './POE/FP/Backbone_IR_50_Epoch_120.pth'


    arcface_fp = IR_50(INPUT_SIZE)

    if os.path.isfile(BACKBONE_RESUME_ROOT_FP):
        arcface_fp.load_state_dict(torch.load(BACKBONE_RESUME_ROOT_FP))
        print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT_FP))

    criterion_id_fp = arcface_fp.to(device)


    BACKBONE_RESUME_ROOT_SS = './POE/SS/Backbone_IR_50_Epoch_100.pth'

    arcface_ss = IR_50(INPUT_SIZE)

    if os.path.isfile(BACKBONE_RESUME_ROOT_SS):
        arcface_ss.load_state_dict(torch.load(BACKBONE_RESUME_ROOT_SS))
        print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT_SS))

    criterion_id_ss = arcface_ss.to(device)


    BACKBONE_RESUME_ROOT_SP = './POE/SP/Backbone_IR_50_Epoch_150.pth'

    arcface_sp = IR_50(INPUT_SIZE)

    if os.path.isfile(BACKBONE_RESUME_ROOT_SP):
        arcface_sp.load_state_dict(torch.load(BACKBONE_RESUME_ROOT_SP))
        print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT_SP))

    criterion_id_sp = arcface_sp.to(device)


    BACKBONE_RESUME_ROOT_PP = './POE/PP/Backbone_IR_50_Epoch_100.pth'


    arcface_pp = IR_50(INPUT_SIZE)

    if os.path.isfile(BACKBONE_RESUME_ROOT_PP):
        arcface_pp.load_state_dict(torch.load(BACKBONE_RESUME_ROOT_PP))
        print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT_PP))

    criterion_id_pp = arcface_ss.to(device)


    fake_gt_csim_list = []
    fake_real_csim_list = []
    real_gt_csim_list = []
    ard_list = []
    fake_files = os.listdir(fake_folder)
    gt_files = os.listdir(gt_folder)
    real_files = os.listdir(real_folder)
    err = 0
    # 遍历两个文件夹中的图像并计算旋转误差
    for fake_file, gt_file, real_file in tqdm(zip(fake_files, gt_files, real_files), desc="Processing"):
        fake_image_path = os.path.join(fake_folder, fake_file)
        gt_image_path = os.path.join(gt_folder, gt_file)
        real_image_path = os.path.join(real_folder, real_file)
        if (fake_file == gt_file):
            # fake_img = Image.open(fake_image_path).convert('RGB').resize((112, 112))
            # gt_img = Image.open(gt_image_path).convert('RGB').resize((112, 112))
            # real_img = Image.open(real_image_path).convert('RGB').resize((112, 112))
            fake_img = cv2.imread(fake_image_path)
            fake_angle = calculate_rotation(fake_img)
            fake_img  = cv2.resize(fake_img, (112, 112))
            gt_img = cv2.imread(gt_image_path)
            gt_angle = calculate_rotation(gt_img)
            gt_img  = cv2.resize(gt_img, (112, 112))
            real_img = cv2.imread(real_image_path)
            real_angle = calculate_rotation(real_img)
            real_img  = cv2.resize(real_img, (112, 112))

            if None in [fake_angle, gt_angle, real_angle]:
                err+=1
                continue

            ard = abs(fake_angle - gt_angle)
            ard_list.append(ard)
            fake_img = convert_tensor(fake_img).to(device).reshape(1,3,112,112)
            gt_img = convert_tensor(gt_img).to(device).reshape(1,3,112,112)
            real_img = convert_tensor(real_img).to(device).reshape(1,3,112,112)


            fake_img = nn.functional.interpolate(fake_img[:, :, :], size=(112, 112), mode='bilinear')
            gt_img = nn.functional.interpolate(gt_img[:, :, :], size=(112, 112), mode='bilinear')
            real_img = nn.functional.interpolate(real_img[:, :, :], size=(112, 112), mode='bilinear')

            criterion_id_ff.eval()
            criterion_id_fs.eval()
            criterion_id_fp.eval()
            criterion_id_ss.eval()
            criterion_id_sp.eval()
            criterion_id_pp.eval()

            with torch.torch.no_grad():
                if abs(real_angle) <= 30 and abs(fake_angle) <= 30:
                    fake_embs = criterion_id_ff(fake_img)
                    gt_embs = criterion_id_ff(gt_img)
                    real_embs = criterion_id_ff(real_img)
                elif abs(real_angle) >= 60 and abs(fake_angle) >= 60:
                    fake_embs = criterion_id_pp(fake_img)
                    gt_embs = criterion_id_pp(gt_img)
                    real_embs = criterion_id_pp(real_img)
                elif (abs(real_angle) >= 30 and abs(real_angle) <= 60) and ( abs(fake_angle) >= 30 and abs(fake_angle) <= 60):
                    fake_embs = criterion_id_ss(fake_img)
                    gt_embs = criterion_id_ss(gt_img)
                    real_embs = criterion_id_ss(real_img)
                elif (abs(real_angle) <= 30 and abs(fake_angle) >= 60) or ( abs(fake_angle) <= 30 and abs(real_angle) >= 60):
                    fake_embs = criterion_id_fp(fake_img)
                    gt_embs = criterion_id_fp(gt_img)
                    real_embs = criterion_id_fp(real_img)
                elif (abs(real_angle) <= 30 and abs(fake_angle) >= 30 and abs(fake_angle) <= 60) or ( abs(fake_angle) <= 30 and abs(real_angle) >= 30 and abs(real_angle) <= 60):
                    fake_embs = criterion_id_fs(fake_img)
                    gt_embs = criterion_id_fs(gt_img)
                    real_embs = criterion_id_fs(real_img)
                else:
                    fake_embs = criterion_id_sp(fake_img)
                    gt_embs = criterion_id_sp(gt_img)
                    real_embs = criterion_id_sp(real_img)

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)

            fake_gt_csim = cos(fake_embs, gt_embs)
            fake_gt_csim_list.append(fake_gt_csim)
            fake_real_csim = cos(fake_embs, real_embs)
            fake_real_csim_list.append(fake_real_csim)
            real_gt_csim = cos(real_embs, gt_embs)
            real_gt_csim_list.append(real_gt_csim)
    print('fake_gt_csim:',sum(fake_gt_csim_list)/len(fake_gt_csim_list))
    print('fake_real_csim:',sum(fake_real_csim_list)/len(fake_real_csim_list))
    print('real_gt_csim:',sum(real_gt_csim_list)/len(real_gt_csim_list))
    print('ard:', sum(ard_list)/len(ard_list))
    print('error count:', err)

fake_image_folder = '/mnt/wwn-0x50014ee0042d9061-part1/RFG_DECA_TRANS/expr/eval/mpie/250000'
gt_image_folder = '/mnt/wwn-0x50014ee0042d9061-part1/RFG_DECA_TRANS/expr/eval/mpie/250000ground_truth'
real_image_folder = '/mnt/wwn-0x50014ee0042d9061-part1/RFG_DECA_TRANS/expr/eval/mpie/250000real'

calculate_csim_for_all_tasks(fake_image_folder, gt_image_folder, real_image_folder)