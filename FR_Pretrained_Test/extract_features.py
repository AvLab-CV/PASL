'''
    implement the feature extractions for light CNN
    @author: Alfred Xiang Wu
    @date: 2017.07.04
'''

from Model.model_irse import IR_50
from Model.model_lightcnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
from Model.resnet50_ft_dims_2048 import resnet50_ft
import torch
import argparse
import pandas as pd
import os
from torch.utils.data import DataLoader
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist
from numpy.linalg import norm
from util.PoseSelect import PoseSelect
from util.LoadPretrained import LoadPretrained
from util.DataLoader import FaceIdPoseDataset
from util.ConcatPath import ConcatPath
from util.InputSize_Select import Transform_Select
import winsound
import torch.nn as nn



parser = argparse.ArgumentParser(description='Eval_SOTA_Model')
# learning & saving parameters
# parser.add_argument('-data-place', type=str, default='D:/Josh2/fnm-master_Validate/test_MPIE/FNM-MB16-Lamda_Fea1000_1_0', help='prepared data path to run program')
# parser.add_argument('-csv-file', type=str, default='../DataList/IJB-A_FOCropped_250_250_84.csv', help='csv file to load image for training')
parser.add_argument('-data-place', type=str,
                    default='D:/face-recognition/FR_Pretrained_Test',
                    help='prepared data path to run program')
parser.add_argument('-csv-file', type=str, default='./DataList/test.csv',
                    help='csv file to load image for training')
parser.add_argument('-model-select', type=str, default='Light_CNN_29', help='Model Select')
parser.add_argument('-cuda', action='store_true', default=True, help='enable the gpu')
parser.add_argument('-batch-size', type=int, default=1, help='batch size for training [default: 8]')
# Evaluation options
# parser.add_argument('-generate-place', type=str, default='D:/Experiment_IJBA_New/Feature/FNM_LightCNN29_Fea15000_FOCropped_250_250_84_epoch7_0', help='prepared data path to run program')
parser.add_argument('-generate-place', type=str,
                    default='D:/04_FaceEvaluation/Experiment_IJBA_Mean_v02/Feature/SA_FNM_MB4_Fea3500_051_Illum_Sym001_All_w_AllGP_1_0',
                    help='prepared data path to run program')
parser.add_argument('-Save-Features', action='store_true', default=True, help='enable the gpu')
parser.add_argument('-Eval-CFP', action='store_true', default=False, help='enable the gpu')

def main():
    global args
    args = parser.parse_args()

    BACKBONE_DICT = {'IR-50': IR_50(112),
                 'Light_CNN_9': LightCNN_9Layers(),
                 'Light_CNN_29': LightCNN_29Layers(),
                 'Light_CNN_29_v2': LightCNN_29Layers_v2(),
                 'VGGFace2': resnet50_ft(weights_path='Pretrained/VGGFace2/resnet50_ft_dims_2048.pth')
    }

    BACKBONE = BACKBONE_DICT[args.model_select]
    Model = LoadPretrained(BACKBONE, args)

    if args.cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Model = Model.to(device)
    Model.eval()
    transforms = Transform_Select(args)
    transformed_dataset = FaceIdPoseDataset(args.csv_file, args.data_place,
                                            transform=transforms)
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False)  # , num_workers=6)

    # Features_List = {'Subject':[], 'Pose':[], 'ImgNum':[], 'Features':[]}
    # count = 0
    # minibatch_size = args.batch_size
    fea = []
    for i, batch_data in enumerate(dataloader):
        if args.model_select == 'VGGFace2': batch_image = (batch_data[0]*255).to(device)
        else: batch_image = batch_data[0].to(device)

        _ = Model(batch_image)
        try: Model_Feature = Model.feature
        except: Model_Feature = Model.module.feature
        fea.append(Model_Feature)
        print(Model_Feature)
    print(len(fea[0]))
    print(len(fea[0][0]))
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    print(cos(fea[0], fea[1]))
    print(cos(fea[0], fea[2]))
    print(cos(fea[0], fea[3]))
    # if args.model == 'LightCNN-9':
    #     model = LightCNN_9Layers(num_classes=args.num_classes)
    # elif args.model == 'LightCNN-29':
    #     model = LightCNN_29Layers(num_classes=args.num_classes)
    # elif args.model == 'LightCNN-29v2':
    #     model = LightCNN_29Layers_v2(num_classes=args.num_classes)
    # else:
    #     print('Error model type\n')
    #
    # model.eval()
    # if args.cuda:
    #     model = torch.nn.DataParallel(model).cuda()

    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         model.load_state_dict(checkpoint['state_dict'])
    # else:
    #     print("=> no checkpoint found at '{}'".format(args.resume))

#     img_list  = read_list(args.img_list)
#     transform = transforms.Compose([transforms.ToTensor()])
#     count     = 0
#     input     = torch.zeros(1, 1, 128, 128)
#     fea = []
#     for img_name in img_list:
#         print(img_name)
#         count = count + 1
#
#
#         img   = cv2.imread(os.path.join(args.root_path, img_name), cv2.IMREAD_GRAYSCALE)
#
#         img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
#         cv2.imshow('2',img)
#         cv2.waitKey(0)
#
#         img   = np.reshape(img, (128, 128, 1))
#
#
#         img   = transform(img)
#         input[0,:,:,:] = img
#
#         start = time.time()
#         if args.cuda:
#             input = input.cuda()
#         with torch.no_grad():
#             # input_var   = torch.autograd.Variable(input, volatile=True)
#             _, features = model(input)
#             end         = time.time() - start
#             print("{}({}/{}). Time: {}".format(os.path.join(args.root_path, img_name), count, len(img_list), end))
#             # save_feature(args.save_path, img_name, features.data.cpu().numpy()[0])
#             fea.append(features)
#
#     from scipy.spatial.distance import cosine
#     print(cosine(fea[0].cpu().detach().numpy(), fea[1].cpu().detach().numpy()))
#
#     cos = nn.CosineSimilarity(dim=1, eps=1e-6)
#     # cos = nn.CosineSimilarity()
#     # print(fea[0].shape)
#     # print(fea[1])
#     print(fea)
#     # print()
#     print(cos(fea[0], fea[1]))
#
#     # input1 = torch.randn(1, 100)
#     # input2 = torch.randn(1, 100)
#     # print(input1.shape)
#     # print(cos(input1, input2))
#
#     # print(fea[0].detach().numpy())
#     # print(cosin_distance(fea[0].numpy(),fea[1].numpy()))
#     # print(cosin_distance(fea[0].numpy(), fea[2].numpy()))
#
# # def cosin_distance(vector1, vector2):
# #     dot_product = 0.0
# #     normA = 0.0
# #     normB = 0.0
# #     for a, b in zip(vector1, vector2):
# #         dot_product += a * b
# #         normA += a ** 2
# #         normB += b ** 2
# #     if normA == 0.0 or normB == 0.0:
# #         return None
# #     else:
# #         return dot_product / ((normA * normB) ** 0.5)
#
#
# def read_list(list_path):
#     img_list = []
#     with open(list_path, 'r') as f:
#         for line in f.readlines()[0:]:
#             img_path = line.strip().split()
#             img_list.append(img_path[0])
#     print('There are {} images..'.format(len(img_list)))
#     return img_list
#
# def save_feature(save_path, img_name, features):
#     img_path = os.path.join(save_path, img_name)
#     img_dir  = os.path.dirname(img_path) + '/';
#     if not os.path.exists(img_dir):
#         os.makedirs(img_dir)
#     fname = os.path.splitext(img_path)[0]
#     fname = fname + '.feat'
#     fid   = open(fname, 'wb')
#     fid.write(features)
#     fid.close()

if __name__ == '__main__':
    main()