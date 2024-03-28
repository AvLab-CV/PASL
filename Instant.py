# coding: utf-8

__author__ = 'TTTT'

import face_alignment
from torchvision import transforms
import torch
import argparse
import imageio
import cv2
import numpy as np
from tqdm import tqdm
import yaml
from collections import deque
from utils.pncc import pncc
import torchvision.utils as vutils
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA 
from utils.render import render
from utils.functions import cv_draw_landmark
from skimage import transform as trans
import torch.nn as nn
from core.model_lm_talking import build_model
from core.checkpoint import CheckpointIO
from os.path import join as ospj
import argparse
import cv2
import numpy as np
import torch

class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        _, self.nets_ema = build_model(args)

        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)


        self.ckptios = [CheckpointIO(ospj(args.checkpoint_dir, '{}_nets_ema.ckpt'.format(args.resume_iter)), **self.nets_ema)]
        self.half() #modify
        self.to(self.device)


        self._load_checkpoint(args.resume_iter)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    @torch.no_grad()

    def sample(self, src, pncc, lm):
        src = src.type(torch.HalfTensor).cuda() #modify
        pncc = pncc.type(torch.HalfTensor).cuda()
        lm = lm.type(torch.HalfTensor).cuda()

        src = src.to(self.device)
        pncc = pncc.to(self.device)
        lm = lm.to(self.device)
        args = self.args

        if args.masks:
            masks = lm
        else:
            masks = None
        x_fake = self.nets_ema.generator(pncc, lm, src, masks=masks)
        img_fake = tensor_to_np(x_fake[0].unsqueeze(0))
        img_fake_2 = tensor_to_np(x_fake[1].unsqueeze(0))
        img_fake_3 = tensor_to_np(x_fake[2].unsqueeze(0))
        img_fake_4 = tensor_to_np(x_fake[3].unsqueeze(0))
        return img_fake, img_fake_2, img_fake_3, img_fake_4

    @torch.no_grad()
    def extract(self, src):

        src = src.to(self.device)

        s_ref = self.nets_ema.style_encoder(src)

        return s_ref

def drawshape(landmarks):
    img = np.zeros((256, 256, 3))
    line_color = (255, 255, 255)
    line_width = 2

    diff = int(float(landmarks[42][0])) - int(float(landmarks[36][0]))

    if diff<=30 and int(float(landmarks[30][0])) > int(float(landmarks[42][0])):
        for n in range(0, 12):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(17, 21):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(27, 30):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(31, 33):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(36, 41):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[36][0])), int(float(landmarks[36][1]))),
                 (int(float(landmarks[41][0])), int(float(landmarks[41][1]))), line_color, line_width)
        for n in range(48, 51):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[51][0])), int(float(landmarks[51][1]))),
                 (int(float(landmarks[62][0])), int(float(landmarks[62][1]))), line_color, line_width)
        for n in range(60, 62):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[60][0])), int(float(landmarks[60][1]))),
                 (int(float(landmarks[67][0])), int(float(landmarks[67][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[67][0])), int(float(landmarks[67][1]))),
                 (int(float(landmarks[66][0])), int(float(landmarks[66][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[57][0])), int(float(landmarks[57][1]))),
                 (int(float(landmarks[66][0])), int(float(landmarks[66][1]))), line_color, line_width)
        for n in range(57, 59):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[48][0])), int(float(landmarks[48][1]))),
                 (int(float(landmarks[59][0])), int(float(landmarks[59][1]))), line_color, line_width)
    elif diff <= 30 and int(float(landmarks[30][0])) < int(float(landmarks[42][0])):
        for n in range(4, 16):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(22, 26):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(27, 30):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(33, 35):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(42, 47):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[42][0])), int(float(landmarks[42][1]))),
                 (int(float(landmarks[47][0])), int(float(landmarks[47][1]))), line_color, line_width)
        for n in range(51, 57):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[51][0])), int(float(landmarks[51][1]))),
                 (int(float(landmarks[62][0])), int(float(landmarks[62][1]))), line_color, line_width)
        for n in range(62, 66):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[66][0])), int(float(landmarks[66][1]))),
                 (int(float(landmarks[57][0])), int(float(landmarks[57][1]))), line_color, line_width)

    else:
        for n in range(0, 16):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(17, 21):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(22, 26):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(27, 30):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(31, 35):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        for n in range(36, 41):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[36][0])), int(float(landmarks[36][1]))),
                 (int(float(landmarks[41][0])), int(float(landmarks[41][1]))), line_color, line_width)
        for n in range(42, 47):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[42][0])), int(float(landmarks[42][1]))),
                 (int(float(landmarks[47][0])), int(float(landmarks[47][1]))), line_color, line_width)
        for n in range(48, 59):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[48][0])), int(float(landmarks[48][1]))),
                 (int(float(landmarks[59][0])), int(float(landmarks[59][1]))), line_color, line_width)
        for n in range(60, 67):
            cv2.line(img, (int(float(landmarks[n][0])), int(float(landmarks[n][1]))),
                     (int(float(landmarks[n + 1][0])), int(float(landmarks[n + 1][1]))), line_color, line_width)
        cv2.line(img, (int(float(landmarks[60][0])), int(float(landmarks[60][1]))),
                 (int(float(landmarks[67][0])), int(float(landmarks[67][1]))), line_color, line_width)
    return img

def tran_point(point, M):
    pts = np.float32(point).reshape([-1,2])
    pts = np.hstack([pts,np.ones([len(pts),1])]).T
    target_point = np.dot(M,pts)
    return target_point.squeeze()


def get_arcface(rimg, shape):
    landmark = []

    for p in shape[0]:
        w, h = p[0], p[1]
        landmark.append([w, h])

    src = np.array([
      [30.2946*2, 51.6963*2],
      [65.5318*2, 51.5014*2],
      [48.0252*2, 71.7366*2],
      [33.5493*2, 92.3655*2],
      [62.7299*2, 92.2041*2] ], dtype=np.float32 )
    src[:,0] += 31.9721

    landmark = np.array(landmark)
    assert landmark.shape[0] == 68 or landmark.shape[0] == 5
    assert landmark.shape[1] == 2
    if landmark.shape[0] == 68:
        landmark5 = np.zeros((5, 2), dtype=np.float32)
        landmark5[0] = (landmark[36] + landmark[39]) / 2
        landmark5[1] = (landmark[42] + landmark[45]) / 2
        landmark5[2] = landmark[30]
        landmark5[3] = landmark[48]
        landmark5[4] = landmark[54]
    else:
        print('5')
        landmark5 = landmark
    tform = trans.SimilarityTransform()
    tform.estimate(landmark5, src)
    M = tform.params[0:2, :]
    img_crop = cv2.warpAffine(rimg, M, (256, 256), borderValue=0.0)

    # for i in range(68):
    #     landmark[i] = tran_point(landmark[i], M)

    # img_shape = drawshape(landmark)


    # LM = []
    # for LM_crop in landmark:

    #     LM.append(LM_crop[0]/256)
    #     LM.append(LM_crop[1]/256)
    return img_crop#, img_shape, LM

def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        img=img
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)

def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img

def turn_eye(fake_lm, x2_label):
    N, C = fake_lm.size()

    real_left_eye_dis = torch.abs(((x2_label[:, 2 * 36:2 * 36 + 1] - x2_label[:, 2 * 39:2 * 39 + 1]) ** 2 + (
            x2_label[:, 2 * 36 + 1:2 * 36 + 2] - x2_label[:, 2 * 39 + 1:2 * 39 + 2]) ** 2) ** (0.5))

    fake_left_eye_dis = torch.abs(((fake_lm[:, 2 * 36:2 * 36 + 1] - fake_lm[:, 2 * 39:2 * 39 + 1]) ** 2 + (
            fake_lm[:, 2 * 36 + 1:2 * 36 + 2] - fake_lm[:, 2 * 39 + 1:2 * 39 + 2]) ** 2) ** (0.5))

    ratio = fake_left_eye_dis/real_left_eye_dis

    real_left_eye_1_x = x2_label[:, 2 * 37:2 * 37 + 1] - x2_label[:, 2 * 41:2 * 41 + 1]
    real_left_eye_1_y = x2_label[:, 2 * 37 + 1:2 * 37 + 2] - x2_label[:, 2 * 41 + 1:2 * 41 + 2]

    real_left_eye_2_x = x2_label[:, 2 * 38:2 * 38 + 1] - x2_label[:, 2 * 40:2 * 40 + 1]
    real_left_eye_2_y = x2_label[:, 2 * 38 + 1:2 * 38 + 2] - x2_label[:, 2 * 40 + 1:2 * 40 + 2]

    real_right_eye_1_x = x2_label[:, 2 * 43:2 * 43 + 1] - x2_label[:, 2 * 47:2 * 47 + 1]
    real_right_eye_1_y = x2_label[:, 2 * 43 + 1:2 * 43 + 2] - x2_label[:, 2 * 47 + 1:2 * 47 + 2]

    real_right_eye_2_x = x2_label[:, 2 * 44:2 * 44 + 1] - x2_label[:, 2 * 46:2 * 46 + 1]
    real_right_eye_2_y = x2_label[:, 2 * 44 + 1:2 * 44 + 2] - x2_label[:, 2 * 46 + 1:2 * 46 + 2]


    for i in range(0, N):
        fake_lm[i, 2 * 37:2 * 37 + 1] = fake_lm[i, 2 * 41:2 * 41 + 1] + real_left_eye_1_x[i]*ratio[i]
        fake_lm[i, 2 * 37 + 1:2 * 37 + 2] = fake_lm[i, 2 * 41 + 1:2 * 41 + 2] + real_left_eye_1_y[i]*ratio[i]

        fake_lm[i, 2 * 38:2 * 38 + 1] = fake_lm[i, 2 * 40:2 * 40 + 1] + real_left_eye_2_x[i]*ratio[i]
        fake_lm[i, 2 * 38 + 1:2 * 38 + 2] = fake_lm[i, 2 * 40 + 1:2 * 40 + 2] + real_left_eye_2_y[i]*ratio[i]

        fake_lm[i, 2 * 43:2 * 43 + 1] = fake_lm[i, 2 * 47:2 * 47 + 1] + real_right_eye_1_x[i]*ratio[i]
        fake_lm[i, 2 * 43 + 1:2 * 43 + 2] = fake_lm[i, 2 * 47 + 1:2 * 47 + 2] + real_right_eye_1_y[i]*ratio[i]

        fake_lm[i, 2 * 44:2 * 44 + 1] = fake_lm[i, 2 * 46:2 * 46 + 1] + real_right_eye_2_x[i]*ratio[i]
        fake_lm[i, 2 * 44 + 1:2 * 44 + 2] = fake_lm[i, 2 * 46 + 1:2 * 46 + 2] + real_right_eye_2_y[i]*ratio[i]
    return fake_lm

def show_map(landmark):
    N, C= landmark.size()
    img = np.ones((N,3,256, 256))
    img =torch.from_numpy(img)
    for i in range(0, N):
        img_3 = np.zeros((256, 256, 3))
        line_color = (255, 255, 255)
        line_width = 2
        lm_x = []
        lm_y = []
        for num in range(68):
            lm_x.append(landmark[i, 2*num]*256)
            lm_y.append(landmark[i, 2*num+1]*256)

        diff_1 = int(float(lm_x[42])) - int(float(lm_x[36]))
        diff_2 = int(float(lm_x[45])) - int(float(lm_x[39]))

        if diff_1<=30 and int(float(lm_x[30])) > int(float(lm_x[42])):
            for n in range(0, 12):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            for n in range(17, 21):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            for n in range(31, 33):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            for n in range(36, 41):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[36])), int(float(lm_y[36]))),
                     (int(float(lm_x[41])), int(float(lm_y[41]))), line_color, line_width)
            for n in range(48, 51):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[51])), int(float(lm_y[51]))),
                     (int(float(lm_x[62])), int(float(lm_y[62]))), line_color, line_width)
            for n in range(60, 62):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[60])), int(float(lm_y[60]))),
                     (int(float(lm_x[67])), int(float(lm_y[67]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[66])), int(float(lm_y[66]))),
                     (int(float(lm_x[67])), int(float(lm_y[67]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[57])), int(float(lm_y[57]))),
                     (int(float(lm_x[66])), int(float(lm_y[66]))), line_color, line_width)
            for n in range(57, 59):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[48])), int(float(lm_y[48]))),
                     (int(float(lm_x[59])), int(float(lm_y[59]))), line_color, line_width)

        elif diff_2 <= 30 and int(float(lm_x[30])) < int(float(lm_x[42])):

            for n in range(4, 16):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            for n in range(22, 26):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            for n in range(27, 30):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            for n in range(33, 35):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            for n in range(42, 47):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[42])), int(float(lm_y[42]))),
                     (int(float(lm_x[47])), int(float(lm_y[47]))), line_color, line_width)
            for n in range(51, 57):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[51])), int(float(lm_y[51]))),
                     (int(float(lm_x[62])), int(float(lm_y[62]))), line_color, line_width)
            for n in range(62, 66):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                         (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[66])), int(float(lm_y[66]))),
                     (int(float(lm_x[57])), int(float(lm_y[57]))), line_color, line_width)

        else:
            for n in range(0, 16):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                            (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            for n in range(17, 21):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                            (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            for n in range(22, 26):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                            (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            for n in range(27, 30):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                            (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))),line_color, line_width)
            for n in range(31, 35):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                            (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            for n in range(36, 41):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                            (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[36])), int(float(lm_y[36]))),
                        (int(float(lm_x[41])), int(float(lm_y[41]))), line_color, line_width)
            for n in range(42, 47):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                            (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[42])), int(float(lm_y[42]))),
                        (int(float(lm_x[47])), int(float(lm_y[47]))),line_color, line_width)
            for n in range(48, 59):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                            (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[48])), int(float(lm_y[48]))),
                        (int(float(lm_x[59])), int(float(lm_y[59]))), line_color, line_width)
            for n in range(60, 67):
                cv2.line(img_3, (int(float(lm_x[n])), int(float(lm_y[n]))),
                            (int(float(lm_x[n + 1])), int(float(lm_y[n + 1]))), line_color, line_width)
            cv2.line(img_3, (int(float(lm_x[60])), int(float(lm_y[60]))),
                        (int(float(lm_x[67])), int(float(lm_y[67]))), line_color, line_width)
        tensor = transforms.ToTensor()(img_3)
        img[i,:,:,:] = tensor
    # img = img.type(torch.cuda.FloatTensor)
    img = img.type(torch.HalfTensor).cuda() #modify
    return img

def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def save_image(x, ncol, filename, loss = 'perceptual'):
    if loss == 'arcface':
        x = denormalize(x)
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)

def main(args):
    solver = Solver(args)
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    print("Loading the FAN Model......")
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device='cuda')
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

    # Given a camera
    # before run this line, make sure you have installed `imageio-ffmpeg`
    reader = imageio.get_reader("<video0>")
    n_pre, n_next = args.n_pre, args.n_next
    n = n_pre + n_next + 1
    queue_ver = deque()
    queue_frame = deque()

    # run
    dense_flag = args.opt in ('2d_dense', '3d', 'pncc')
    pre_ver = None
    
    #input img
    # img_crop = cv2.imread('./examples/663.jpg')
    # img_crop_2 = cv2.imread('./examples/287.jpg')
    # img_crop_3 = cv2.imread('./examples/96.jpg')
    # img_crop_4 = cv2.imread('./examples/483.jpg')
    img_crop = cv2.imread('./examples/663.jpg')
    img_crop_2 = cv2.imread('./examples/189.jpg')
    img_crop_3 = cv2.imread('./examples/287.jpg')
    img_crop_4 = cv2.imread('./examples/96.jpg')

    img_crop = cv2.resize(img_crop, (256, 256))
    img_crop_2 = cv2.resize(img_crop_2, (256, 256))
    img_crop_3 = cv2.resize(img_crop_3, (256, 256))
    img_crop_4 = cv2.resize(img_crop_4, (256, 256))

    source = toTensor(img_crop)
    source_2 = toTensor(img_crop_2)
    source_3 = toTensor(img_crop_3)
    source_4 = toTensor(img_crop_4)

    source_all = np.zeros((4, 3, 256, 256))
    source_all = torch.from_numpy(source_all)
    source_all= source_all.float()

    source_all[0,:,:,:] = source
    source_all[1,:,:,:] = source_2
    source_all[2,:,:,:] = source_3
    source_all[3,:,:,:] = source_4
    source_all=source_all.type(torch.HalfTensor).cuda() #modify

    source_style_code = solver.extract(source_all)
    i = 0
    for i, frame in tqdm(enumerate(reader)):
        frame_bgr = frame[..., ::-1]  # RGB->BGR
        preds = fa.get_landmarks(frame_bgr)
        try:
            img_crop2, img_shape, shape = get_arcface(frame_bgr, preds)
        except:
            continue

        if i == 0:
            # the first frame, detect face, here we only use the first face, you can change depending on your need
            boxes = face_boxes(img_crop2)
            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(img_crop2, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # refine
            param_lst, roi_box_lst = tddfa(img_crop2, [ver], crop_policy='landmark')
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            
            # padding queue
            for _ in range(n_pre):
                queue_ver.append(ver.copy())
            queue_ver.append(ver.copy())

            for _ in range(n_pre):
                queue_frame.append(img_crop2.copy())
            queue_frame.append(img_crop2.copy())
        else:
            try:
                param_lst, roi_box_lst = tddfa(img_crop2, [pre_ver], crop_policy='landmark')
            except:continue

            roi_box = roi_box_lst[0]
            # todo: add confidence threshold to judge the tracking is failed
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(img_crop2)
                boxes = [boxes[0]]
                param_lst, roi_box_lst = tddfa(img_crop2, boxes)

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            queue_ver.append(ver.copy())
            queue_frame.append(img_crop2.copy())

        pre_ver = ver  # for tracking

        # smoothing: enqueue and dequeue ops
        if len(queue_ver) >= n:
            ver_ave = np.mean(queue_ver, axis=0)

            if args.opt == '2d_sparse':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)  # since we use padding
            elif args.opt == '2d_dense':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
            elif args.opt == '3d':
                img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
            elif args.opt == 'pncc':
                img_draw = pncc(queue_frame[n_pre], [ver_ave], tddfa.tri, show_flag=None, wfp=None, with_bg_flag=False)
            else:
                raise ValueError(f'Unknown opt {args.opt}')
            
            
            img_shape = toTensor(img_shape)
            lm_shape = np.zeros((4, 3, 256, 256))
            lm_shape = torch.from_numpy(lm_shape)
            lm_shape= lm_shape.float()

            lm_shape[0,:,:,:] = img_shape
            lm_shape[1,:,:,:] = img_shape
            lm_shape[2,:,:,:] = img_shape
            lm_shape[3,:,:,:] = img_shape
            img_draw = toTensor(img_draw)

            pncc_map = np.zeros((4, 3, 256, 256))
            pncc_map = torch.from_numpy(pncc_map)
            pncc_map= pncc_map.float()

            pncc_map[0,:,:,:] = img_draw
            pncc_map[1,:,:,:] = img_draw
            pncc_map[2,:,:,:] = img_draw
            pncc_map[3,:,:,:] = img_draw
            img_fake, img_fake_2, img_fake_3, img_fake_4 = solver.sample(source_style_code, pncc_map, lm_shape)
            #x = torch.cat((img_draw, img_shape), dim=1)
            #torch.concat((x, lm), dim=1)
            img_fake = cv2.cvtColor(img_fake, cv2.COLOR_RGB2BGR)
            img_fake_2 = cv2.cvtColor(img_fake_2, cv2.COLOR_RGB2BGR)
            img_fake_3 = cv2.cvtColor(img_fake_3, cv2.COLOR_RGB2BGR)
            img_fake_4 = cv2.cvtColor(img_fake_4, cv2.COLOR_RGB2BGR)

            all = np.zeros((516, 816, 3), np.uint8)
            img_crop2 = cv2.resize(img_crop2, (256,256))

            all[130:386, :256, :] = img_crop2 # ref
            img_fake = cv2.resize(img_fake, (256,256))
            img_fake_2 = cv2.resize(img_fake_2, (256,256))
            img_fake_3 = cv2.resize(img_fake_3, (256,256))
            img_fake_4 = cv2.resize(img_fake_4, (256,256))

            all[:256, 300:556, :] = img_fake
            all[:256, 560:816, :] = img_fake_2
            all[260:516, 300:556, :] = img_fake_3
            all[260:516, 560:816, :] = img_fake_4
            #==========================================================
          
            # out.write(all)
            all = cv2.resize(all, (2460,1540))
            cv2.namedWindow("Full-Pose", cv2.WINDOW_AUTOSIZE)
            cv2.resizeWindow("Full-Pose",2460,1540)
            cv2.imshow("Full-Pose", all)

            k = cv2.waitKey(20)
            if (k & 0xff == ord('q')):
                break
            queue_ver.popleft()
            queue_frame.popleft()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The smooth demo of webcam of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-m', '--mode', default='gpu', type=str, help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='pncc', choices=['2d_sparse', '2d_dense', '3d', 'pncc'])
    parser.add_argument('-n_pre', default=1, type=int, help='the pre frames of smoothing')
    parser.add_argument('-n_next', default=1, type=int, help='the next frames of smoothing')
    parser.add_argument('--onnx', action='store_true', default=True)
    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--id_embed_dim', type=int, default=4096,
                        help='ID code dimension by VGGFace')
    parser.add_argument('--id_embed', action='store_true', default=False,
                        help='multi input') 
    #style code dimension
    parser.add_argument('--style_dim', type=int, default=128,
                        help='Style code dimension')
    #loss select
    parser.add_argument('--loss', type=str, default='perceptual',
                        help='the type of loss. [perceptual]')
    #dataset select
    parser.add_argument('--pix2pix', action='store_true', default=False,
                        help='use pix2pix loss')
    parser.add_argument('--multi_discriminator', action='store_true', default=True,
                        help='use multi_discriminator')
    parser.add_argument('--masks', action='store_true', default=False,
                        help='use mask injection')
    parser.add_argument('--self_att', action='store_true', default=False,
                        help='use self-attention')
    parser.add_argument('--w_hpf', type=float, default=1,
                        help='weight for high-pass filtering')
    parser.add_argument('--resume_iter', type=int, default=806000,
                        help='Iterations to resume training/testing')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory for saving network checkpoints')
    args = parser.parse_args()

if __name__ == '__main__':
    main(args)