import sys
import argparse
import cv2
import yaml
import glob
import torch
import os
import numpy as np

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
from utils.tddfa_util import str2bool
from math import cos, sin, atan2, asin, sqrt
import math


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

def calc_pose(param, args):
    P = param[:12].reshape(3, -1).copy()  # camera matrix
    s, R, t3d = P2sRt(P)
    angle = matrix2angle(R)

    # need to be radius
    if args.pose_select==None:
        yaw, pitch, roll = angle
    else:
        yaw = args.pose_select * math.pi / 180
        pitch = 0
        roll = 0
    rotation_matrix = angle2matrix([-yaw, pitch, roll]) * s
    Camera_Matrix = np.concatenate((rotation_matrix, t3d.reshape(3, -1)), axis=1)  # without scale
    param[:12] = Camera_Matrix.reshape(-1, )

    return angle

def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    for files in [args.img_fp, args.img_fp_gt]:
        import os
        # Given a still image path and load to BGR channel
        img = cv2.imread(files)

        # Detect faces, get 3DMM params and roi boxes
        boxes = face_boxes(img)
        n = len(boxes)
        if n == 0:
            print(f'No face detected, exit')
            sys.exit(-1)
        #print(f'Detect {n} faces')

        param_lst, roi_box_lst = tddfa(img, boxes)

        # Visualization and serialization
        dense_flag = args.opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
        old_suffix = get_suffix(files)
        new_suffix = f'.{args.opt}' if args.opt in ('ply', 'obj') else '.jpg'

        wfp = f'result/{files.split("/")[-1].replace(old_suffix, "")}_{args.opt}' + new_suffix
        
        if not os.path.exists('./result'):
            os.mkdir('./result')

        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)

        if args.opt == '2d_sparse':
            draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp, files=files)

        else:
            raise ValueError(f'Unknown opt {args.opt}')

    
    # for file_path in paths:
    #     print(file_path)
    #     data = file_path.split('/')[-1].split('.')[0]
    #     with open(file_path,'r') as p:
    #         det = []
    #         landmark_x = []
    #         landmark_y = []
    #         roi_boxes = []
    #         landmarks = []
    #         for line in p.readlines():
    #             line = line.strip('\n')
    #             det.append(line)

    #         for landmark in det:
    #             x = float(landmark.split(' ')[0])
    #             y = float(landmark.split(' ')[1])
    #             landmark_x.append(x)
    #             landmark_y.append(y)
    #         x_max = max(landmark_x)
    #         x_min = min(landmark_x)
    #         y_max = max(landmark_y)
    #         y_min = min(landmark_y)
    #         bbox = [x_min, y_min, x_max, y_max]

    #         det_bbox = [landmark_x, landmark_y]
    #         roi_boxes.append(bbox)
    #         landmarks.append(det_bbox)
    #     # Save bbox information to a .npy file
    #     np.save('./landmarks_jeff/{}.roi_box'.format(data),roi_boxes)
    #     np.save('./landmarks_jeff/{}.pts_68'.format(data),landmarks)
    #     p.close()
    
    #calculate NME

    # ground-truth
    pose_all_ori = np.load('./pose/482000gt_Pose.npy')

    # reannonated
    pose_all_re = np.load('./pose/482000_Pose.npy')
    print(pose_all_ori)
    def calc_pose(pts68_fit_all, option='ori'):
        if option == 'ori':
            pts68_all = pose_all_ori
        elif option == 're':
            pts68_all = pose_all_re
        
        P = param[:12].reshape(3, -1).copy()  # camera matrix
        s, R, t3d = P2sRt(P)
        angle = matrix2angle(R)

        # need to be radius
        if args.pose_select==None:
            yaw, pitch, roll = angle
        else:
            yaw = args.pose_select * math.pi / 180
            pitch = 0
            roll = 0
        rotation_matrix = angle2matrix([-yaw, pitch, roll]) * s
        Camera_Matrix = np.concatenate((rotation_matrix, t3d.reshape(3, -1)), axis=1)  # without scale
        param[:12] = Camera_Matrix.reshape(-1, )

        return angle
    
    def calc_nme(pts68_fit_all, option='ori'):
        if option == 'ori':
            pts68_all = pose_all_ori
        elif option == 're':
            pts68_all = pts68_all_re

        nme_list = []

        for i in range(len(roi_boxs)):
            pts68_fit = pts68_fit_all[i]
            pts68_gt = pts68_all[i]

            # build bbox
            minx, maxx = np.min(pts68_gt[0, :]), np.max(pts68_gt[0, :])
            miny, maxy = np.min(pts68_gt[1, :]), np.max(pts68_gt[1, :])
            llength = sqrt((maxx - minx) * (maxy - miny))

            #
            dis = pts68_fit - pts68_gt[:2, :]
            dis = np.sqrt(np.sum(np.power(dis, 2), 0))
            dis = np.mean(dis)
            nme = (dis / llength)*100
            nme_list.append(nme)

        nme_list = np.array(nme_list, dtype=np.float32)
        
        print(nme)
        return nme_list

    calc_nme(pts68_all_re, option = 'ori')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str, default = None)
    parser.add_argument('-f_gt','--img_fp_gt', type = str, default = None, help = 'ground-truth image')
    parser.add_argument('-m', '--mode', type=str, default='gpu', help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse')
    parser.add_argument('--show_flag', type=str2bool, default='true', help='whether to show the visualization result')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
    