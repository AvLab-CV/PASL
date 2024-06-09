"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import cv2

import os, sys
from time import time
from scipy.io import savemat
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points
from decalib.datasets import detectors


def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames

class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root[0])
        self.samples.sort()

        self.samples_2 = listdir(root[1])
        self.samples_2.sort()

        self.samples_3 = listdir(root[2])
        self.samples_3.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        fname_2 = self.samples_2[index]
        fname_3 = self.samples_3[index]


        img = Image.open(fname).convert('RGB')
        img_2 = Image.open(fname_2).convert('RGB')
        img_3 = Image.open(fname_3).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img_2 = self.transform(img_2)
            img_3 = self.transform(img_3)
        return img, img_2, img_3

    def __len__(self):
        return len(self.samples)

class LMDataset(data.Dataset):
    def __init__(self, root, transform=None, train_data = 'mpie',multi=False,test='train'):
        self.device = 'cuda'
        self.deca = DECA(config = deca_cfg, device=self.device)
        self.face_detector = detectors.FAN()
        self.multi = multi
        self.test = test
        if multi:

            self.train_data = train_data
            self.transform = transform
            self.targets = None
            self.samples = []
            self.samples2 = []
            self.samples3 = []
            self.samples4 = []
            self.samples5 = []
            self.samples6 = []
            self.samples7 = []
            self.samples8 = []
            self.samples9 = []

            with open(root) as F:

                for line in F:
                    line = line.strip('\n')
                    # print(line.split(' '))
                    self.samples.append(line.split(' ')[0])
                    self.samples2.append(line.split(' ')[1])
                    self.samples3.append(line.split(' ')[2])
                    self.samples4.append(line.split(' ')[3])
                    self.samples5.append(line.split(' ')[4])
                    self.samples6.append(line.split(' ')[5])
                    self.samples7.append(line.split(' ')[6])
                    self.samples8.append(line.split(' ')[7])
                    self.samples9.append(line.split(' ')[8])

        else:

            self.train_data = train_data
            if self.train_data == 'rafd':

                if self.test == 'train':
                    self.transform = transform
                    self.targets = None
                    self.samples = []
                    self.samples2 = []
                    with open(root) as F:
                        for line in F:
                            line = line.strip('\n')
                            self.samples.append(line.split(' ')[0])
                            self.samples2.append(line.split(' ')[1])

                else:
                    self.transform = transform
                    self.targets = None
                    self.samples = []
                    self.samples2 = []
                    self.samples3 = []
                    with open(root) as F:
                        for line in F:
                            line = line.strip('\n')
                            self.samples.append(line.split(' ')[0])
                            self.samples2.append(line.split(' ')[1])
                            #self.samples3.append(line.split(' ')[2])

            else:
                self.transform = transform
                self.targets = None
                self.samples = []
                self.samples_angle=[]
                self.samples2 = []
                self.samples2_angle=[]
                self.samples3 = []
                with open(root) as F:
                    n=0
                    try:
                        for line in F:
                            
                            line = line.strip('\n')
                            # print(line.split(' '))
                            self.samples.append(line.split(' ')[0])
                            self.samples_angle.append(int(line.split(' ')[0].split('/')[-1].split('_')[2]))
                            self.samples2.append(line.split(' ')[1])
                            self.samples2_angle.append(int(line.split(' ')[1].split('/')[-1].split('_')[2]))
                            try:
                                gt_path = line.split(' ')[2]
                                self.samples3.append(gt_path)
                            except:
                                self.samples3.append(line.split(' ')[1])
                           
                    except Exception as e:
                        n+=1
                        print('error {} images'.format(n))
                        # print(self.samples[-1])
                        # print(self.samples2[-1])
                        del self.samples[-1]
                        del self.samples2[-1]
                        pass



    def __getitem__(self, index):
        if self.multi:
            fname_folder = self.samples[index]
            fname2_folder = self.samples2[index]
            fname3_folder = self.samples3[index]
            fname4_folder = self.samples4[index]
            fname5_folder = self.samples5[index]
            fname6_folder = self.samples6[index]
            fname7_folder = self.samples7[index]
            fname8_folder = self.samples8[index]
            fname9_folder = self.samples9[index]







            if self.train_data == 'vox1':


                fname = fname_folder
                fname2 = fname2_folder
                fname3 = fname3_folder
                fname4 = fname4_folder
                fname5 = fname5_folder
                fname6 = fname6_folder
                fname7 = fname7_folder
                fname8 = fname8_folder
                fname9 = fname9_folder


                fname_lm9 = fname9.split('unzippedFaces')[0] + 'lm/unzippedFaces' + fname9.split('unzippedFaces')[1]



            img = Image.open(fname).convert('RGB')
            img2 = Image.open(fname2).convert('RGB')
            img3 = Image.open(fname3).convert('RGB')
            img4 = Image.open(fname4).convert('RGB')
            img5 = Image.open(fname5).convert('RGB')

            img6 = Image.open(fname6).convert('RGB')
            img7 = Image.open(fname7).convert('RGB')
            img8 = Image.open(fname8).convert('RGB')
            img9 = Image.open(fname9).convert('RGB')

            img_lm9 = Image.open(fname_lm9).convert('RGB')

            if self.transform is not None:
                img = self.transform(img)
                img2 = self.transform(img2)
                img3 = self.transform(img3)
                img4 = self.transform(img4)
                img5 = self.transform(img5)
                img6 = self.transform(img6)
                img7 = self.transform(img7)
                img8 = self.transform(img8)
                img9 = self.transform(img9)

                img_lm9 = self.transform(img_lm9)
                # img_lm = self.transform(img_lm)

            return img, img2, img3, img4,img5,img6,img7,img8, img9, img_lm9

        else:

            fname_folder = self.samples[index]
            fname2_folder = self.samples2[index]
            fname3_folder = self.samples3[index]
            fname_angle = self.samples_angle[index]
            fname2_angle = self.samples2_angle[index]

            if self.train_data == 'mpie':
                

                fname = fname_folder
                fname2 = fname2_folder
                fname3 = fname3_folder
                name_angle = fname_angle
                name2_angle = fname2_angle
                
                # fname = fname.replace('depths_256', 'crop_256')
                # fname2 = fname2.replace('depths_256', 'crop_256')

                # lm = fname2.split('crop_256')[0] + 'LM_256' +fname2.split('crop_256')[1]
                # pncc = fname2.split('crop_256')[0] + 'depths_256' +fname2.split('crop_256')[1]


            elif self.train_data == 'rafd':
                fname = fname_folder
                fname2 = fname2_folder
                fname = fname.split('\t')[0].replace('\\','/')
                fname2 = fname2.replace('\\','/')
                
                lm = fname2.split('rafd_crop_256')[0] + 'rafd_LM_256' + fname.split('rafd_crop_256')[1]
                pncc = fname2.split('rafd_crop_256')[0] + 'rafd_LM_256' + fname2.split('rafd_crop_256')[1]

                # if self.test == 'train':
                #     fname = fname_folder
                #     fname2 = fname2_folder
                #     # fname_lm = fname.split('rafd_crop_256')[0] + 'rafd_LM_256' + fname.split('rafd_crop_256')[1]
                #     # fname_lm2 = fname2.split('rafd_crop_256')[0] + 'rafd_LM_256' + fname2.split('rafd_crop_256')[1]
                #     fname_lm = fname.split('rafd_crop_256')[0] + 'rafd_pncc_256' + fname.split('rafd_crop_256')[1]
                #     fname_lm2 = fname2.split('rafd_crop_256')[0] + 'rafd_pncc_256' + fname2.split('rafd_crop_256')[1]
                #     #print(fname_lm)
                # else:
                #     fname = fname2_folder
                #     fname2 = self.samples3[index]
                #     fname_lm = fname_folder
                #     fname_lm2 = fname_folder


            elif self.train_data == 'vox1':

                fname = fname_folder
                fname2 = fname2_folder
                
                fname = fname.split('\t')[0].replace('\\','/')
               
                fname2 = fname2.replace('\\','/')

                lm = fname2.split('vox1_full_face_crop_256')[0] + 'LM_256' + fname2.split('vox1_full_face_crop_256')[1]
                # pncc = '/media/avlab/2tb/RFG_pncc_landmark/new_Vox2/pncc_256' + fname2.split('crop_256')[1]
                # pncc = fname2.split('crop_256')[0] + 'pncc_256' + fname2.split('crop_256')[1]
            elif self.train_data == 'vox2':

                fname = fname_folder
                fname2 = fname2_folder
                fname3 = fname3_folder
                name_angle = fname_angle
                name2_angle = fname2_angle
                # fname = fname.replace('\\','/') #pncc

                # fname2 = fname2.replace('pncc_256','crop_256') #pncc
                # pncc = fname2.split('crop_256')[0] + 'pncc_256' + fname2.split('crop_256')[1]
                lm = fname2.split('crop_256')[0] + 'LM_256' + fname2.split('crop_256')[1]
                # pncc = fname2.split('crop_256')[0] + 'depth_256' + fname2.split('crop_256')[1]
                # lm = fname2.split('crop_256')[0] + 'depth_256' + fname2.split('crop_256')[1]
            # sor = fname.split(",")[0]
            # flow = fname.split(",")[1]
            # ref = fname2


            img = Image.open(fname).convert('RGB')
            # img_lm = Image.open(pncc).convert('RGB')
            img2 = Image.open(fname2).convert('RGB')
            gt = Image.open(fname3).convert('RGB')
            # img_lm2 = Image.open(pncc).convert('RGB')
            # lm = Image.open(lm).convert('RGB')
            img_lm2, lm = get_depth_render(self.deca, self.face_detector, fname, fname2)
            img_lm2 = img_lm2.resize((256,256))
            img_lm = img_lm2
            # img_lm2 = img_lm2

            flattened_width= img_lm2.width-20
            flattened_width1= lm.width-20
            # flattened_height= img_lm2.height+20
            # 壓扁照片
            new_image = Image.new(img_lm2.mode, (flattened_width, img_lm2.height))
            new_image1= Image.new(lm.mode, (flattened_width1, lm.height))
            # 複製原始圖像到新圖像的中間，加上 margin
            new_image.paste(img_lm2, (0, 10))
            new_image1.paste(lm, (0, 10))
            # 將新圖像重新調整為目標尺寸
            # flattened_image = img_lm2 .resize(( flattened_width, flattened_height))    
            # plt.imshow(flattened_image)    
            # flattened_image1 = lm .resize(( flattened_width, img_lm2.height))     
            # 將壓扁後的照片重新調整為目標尺寸
            img_lm2=new_image.resize((224,224),Image.ANTIALIAS)

            lm=new_image1.resize((256,256),Image.ANTIALIAS)

            if self.transform is not None:


                img = self.transform(img)
                img2 = self.transform(img2)
                img_lm2 = self.transform(img_lm2)
                img_lm = self.transform(img_lm)
                lm = self.transform(lm)
                gt = self.transform(gt)
            return img, img2, img_lm, img_lm2, lm, gt, torch.tensor(name_angle),torch.tensor(name2_angle)
            # return img, img2, img_lm, img_lm2, lm



    def __len__(self):
        return len(self.samples)


def get_depth_render(deca, face_detector, src_path, ref_path):
    testdata = datasets.TestData(face_detector, [src_path, ref_path], iscrop=True, face_detector='fan', sample_step=10)
    device = 'cuda'
    i=0
    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = 'pytorch3d'
    deca_cfg.model.extract_tex = True
    src_name = testdata[i]['imagename']
    src = testdata[i]['image'].to(device)[None,...]
    ref_name = testdata[i+1]['imagename']
    ref = testdata[i+1]['image'].to(device)[None,...]
    with torch.no_grad():
        codedict1 = deca.encode(src)
        codedict2 = deca.encode(ref)
        src_shape = codedict1['shape']

        light_code = codedict1['light']
        tex_code = codedict1['tex']
        detail_code = codedict1['detail']

        ref_shape = codedict2['shape']
        temp = codedict2
        temp['shape'] = src_shape
        temp['light'] = light_code
        temp['tex'] = tex_code
        temp['detail'] = detail_code
        tform = testdata[i+1]['tform'][None, ...]
        tform = torch.inverse(tform).transpose(1,2).to(device)
        original_image = testdata[i+1]['original_image'][None, ...].to(device)
        orig_opdict, orig_visdict = deca.decode(temp, render_orig=True, original_image=original_image, tform=tform)    
        orig_visdict['inputs'] = original_image
        # cv2.imwrite('1.png', cv2.resize(util.tensor2image(orig_visdict['landmarks2d'][0]),(256,256)))
        lm_image = cv2.resize(util.tensor2image(orig_visdict['landmarks2d'][0]),(256,256))
        depth_image = deca.render.render_depth(orig_opdict['trans_verts']).repeat(1,3,1,1)[0]

        depth_image = depth_image.detach().cpu().numpy()
        depth_image = depth_image*255.
        depth_image = np.maximum(np.minimum(depth_image, 255), 0)
        depth_image = depth_image.transpose(1,2,0)[:,:,[2,1,0]]
        depth_image = Image.fromarray(np.uint8(depth_image))
        lm_image = Image.fromarray(lm_image)
        
    return depth_image, lm_image


def get_eval_loader_vgg(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=False,
                    num_workers=0, drop_last=False, train_data = 'mpie',multi=False, mode='train'):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299

    else:
        height, width = img_size, img_size


    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor()

    ])



    dataset = LMDataset(root, transform=transform, train_data = train_data,multi=multi,test=mode)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader_vgg(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4, train_data='mpie',multi=False, mode='train'):
    print('Preparing DataLoader for the generation phase...')

    transform = transforms.Compose([
        
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor()
    ])


    dataset = LMDataset(root, transform=transform, train_data = train_data,multi=multi,test=mode)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           pin_memory=True)

def get_eval_loader_2(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=False,
                    num_workers=4, drop_last=False, train_data = 'mpie',multi=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


    dataset = DefaultDataset(root, transform=transform)
    # dataset = LMDataset(root, transform=transform, train_data = train_data)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


class InputFetcher:
    def __init__(self, loader, latent_dim=16, mode='', multi=False):
        self.loader = loader
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
        self.multi = multi

    def _fetch_inputs(self):
        if self.multi:
            try:
                x1, x2, x3, x4, x5, x6, x7, x8, x9, x9_lm = next(self.iter)
            except (AttributeError, StopIteration):
                self.iter = iter(self.loader)
                x1, x2, x3, x4, x5, x6, x7, x8, x9, x9_lm = next(self.iter)
            return x1, x2, x3, x4, x5, x6, x7, x8, x9, x9_lm
        else:
            try:
                x1, x2, x_lm, x2_lm, lm , gt, x1_angle,x2_angle= next(self.iter)
                # x1, x2, x_lm, x2_lm, lm = next(self.iter)
            except (AttributeError, StopIteration):
                self.iter = iter(self.loader)
                x1, x2, x_lm, x2_lm, lm, gt, x1_angle,x2_angle= next(self.iter)
                # x1, x2, x_lm, x2_lm, lm = next(self.iter)
            return x1, x2, x_lm, x2_lm, lm, gt, x1_angle, x2_angle
            # return x1, x2, x_lm, x2_lm, lm




    def __next__(self):
        if self.multi:
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x9_lm = self._fetch_inputs()

            inputs = Munch(x1=x1, x2=x2, x3=x3, x4=x4, x5=x5, x6=x6, x7=x7, x8=x8, x9=x9, x9_lm=x9_lm)

            return Munch({k: v.to(self.device)
                          for k, v in inputs.items()})
        else:
            x1, x2, x_lm, x2_lm, lm, gt, x1_angle,x2_angle= self._fetch_inputs() #poe
            # x1, x2, x_lm, x2_lm, lm= self._fetch_inputs()

            inputs = Munch(x1=x1, x2=x2, x_lm=x_lm, x2_lm=x2_lm, lm=lm, gt=gt, x1_angle=x1_angle,x2_angle=x2_angle)
            # inputs = Munch(x1=x1, x2=x2, x_lm=x_lm, x2_lm=x2_lm, lm=lm)

            return Munch({k: v.to(self.device)
                          for k, v in inputs.items()})
