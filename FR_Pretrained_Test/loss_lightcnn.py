'''
    implement the feature extractions for light CNN
    @author: Alfred Xiang Wu
    @date: 2017.07.04
'''

from FR_Pretrained_Test.Model.model_irse import IR_50
from FR_Pretrained_Test.Model.model_lightcnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
from FR_Pretrained_Test.Model.resnet50_ft_dims_2048 import resnet50_ft
import torch
import argparse
import pandas as pd
import os
from torch.utils.data import DataLoader
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist
from numpy.linalg import norm
from FR_Pretrained_Test.util.PoseSelect import PoseSelect
from FR_Pretrained_Test.util.LoadPretrained import LoadPretrained
from FR_Pretrained_Test.util.DataLoader import FaceIdPoseDataset
from FR_Pretrained_Test.util.ConcatPath import ConcatPath
from FR_Pretrained_Test.util.InputSize_Select import Transform_Select
#import winsound
import torch.nn as nn



class LossEG(nn.Module):
    def __init__(self, feed_forward=True, gpu=None):
        super(LossEG, self).__init__()

        print('load model')
        self.LIGHTCNN_FACE_AC = Lightcnn_Activations(lightcnn(pretrained=True), [1, 6, 11, 18, 25])
        print('finished load lightcnn model')
        assert False


        # self.match_loss = not feed_forward
        self.gpu = gpu
        if gpu is not None:
            self.cuda(gpu)

    def loss_cnt(self, x, x_hat):
        # IMG_NET_MEAN = torch.Tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).to(x.device)
        # IMG_NET_STD = torch.Tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).to(x.device)

        # x = (x - IMG_NET_MEAN) / IMG_NET_STD
        # x_hat = (x_hat - IMG_NET_MEAN) / IMG_NET_STD


        # VGG Face Loss
        lightcnn_x_hat = self.LIGHTCNN_FACE_AC(x_hat)
        lightcnn_x = self.LIGHTCNN_FACE_AC(x)

        lightcnn_loss = 0
        for i in range(0, len(lightcnn_x)):
            lightcnn_loss += F.l1_loss(lightcnn_x_hat[i], lightcnn_x[i])

        return lightcnn_loss


    def forward(self, x, x_hat):
        if self.gpu is not None:
            x = x.cuda(self.gpu)
            x_hat = x_hat.cuda(self.gpu)


        cnt = self.loss_cnt(x, x_hat)


        return cnt .reshape(1)

class Lightcnn_Activations(nn.Module):
    """
    This class allows us to execute only a part of a given VGG network and obtain the activations for the specified
    feature blocks. Note that we are taking the result of the activation function after each one of the given indeces,
    and that we consider 1 to be the first index.
    """
    def __init__(self, lightcnn, feature_idx):
        super(Lightcnn_Activations, self).__init__()
        print(lightcnn.module)
        assert False
        features = list(lightcnn.features)
        self.features = nn.ModuleList(features).eval()
        self.idx_list = feature_idx

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            # print(x.shape)
            if ii in self.idx_list:
                results.append(x)

        return results

def lightcnn(pretrained=False, **kwargs):
    BACKBONE_RESUME_ROOT = 'D:/face-reenactment/stargan-v2-master/FR_Pretrained_Test/Pretrained/LightCNN/LightCNN_29Layers_V2_checkpoint.pth.tar'
    if pretrained:
        kwargs['init_weights'] = False
    model = LightCNN_29Layers_v2()
    if pretrained:
        model = WrappedModel(model)
        checkpoint = torch.load(BACKBONE_RESUME_ROOT)
        model.load_state_dict(checkpoint['state_dict'])
    print(model)
    return model

class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module
    def forward(self, x):
        return self.module(x)