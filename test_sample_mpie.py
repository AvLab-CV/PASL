"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse
from torch.backends import cudnn
import torch
from core.solver_lm_perceptual import Solver


def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    solver = Solver(args)
    print('eval')
    if args.mode == 'eval':
        solver.evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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
                        help='Style code dimension')   #128

    #loss select
    parser.add_argument('--loss', type=str, default='perceptual',
                        help='the type of loss. [perceptual]')
    #dataset select
    parser.add_argument('--dataset', type=str, default='mpie',
                        help='')

    parser.add_argument('--pix2pix', action='store_true', default=False,
                        help='use pix2pix loss')

    parser.add_argument('--multi_discriminator', action='store_true', default=True,
                        help='use multi_discriminator')
    parser.add_argument('--style_cyc', action='store_true', default=True,
                        help='use style consistency loss')
    parser.add_argument('--arc', action='store_true', default=False,
                        help='use arcface loss')

    parser.add_argument('--meg', action='store_true', default=False,
                        help='use arcface loss')
    parser.add_argument('--shape_loss', action='store_true', default=True,
                        help='use shape_id loss')

    parser.add_argument('--config', type=str, default='configs/mb1_120x120.yml')

    parser.add_argument('--multi', action='store_true', default=False,
                        help='multi input')

    parser.add_argument('--masks', action='store_true', default=False,
                        help='use mask injection')
    parser.add_argument('--self_att', action='store_true', default=True,
                        help='use self-attention')

    parser.add_argument('--landmark_loss', type=float, default=False,
                        help='use landmark l1 loss')
    parser.add_argument('--vgg19_recon_loss', type=float, default=True,
                        help='use landmark l1 loss')

    # weight for objective functions

    parser.add_argument('--lambda_pixel', type=float, default=1,
                        help='Weight for pixel l1 loss')

    parser.add_argument('--lambda_fm', type=float, default=1,
                        help='Weight for feature match loss')

    parser.add_argument('--lambda_style_cyc', type=float, default=1,
                        help='Weight for id style consistency loss')

    parser.add_argument('--lambda_id', type=float, default=1,
                        help='Weight for id loss')
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')


    parser.add_argument('--ds_iter', type=int, default=10,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=1,
                        help='weight for high-pass filtering')

    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')

    parser.add_argument('--resume_iter', type=int, default=250000,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=16,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=1,
                        help='Number of generated images per domain during sampling')

    # misc
    parser.add_argument('--mode', type=str, required=False,default='eval',
                        choices=['eval'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')
    
    parser.add_argument('--checkpoint_dir', type=str, default='./experiment/mpie',
                        help='Directory for saving network checkpo0ints')

    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='./expr/eval/mpie',
                        help='Directory for saving metrics, i.e., FID and LPIPS')

    args = parser.parse_args()


    if args.dataset == 'mpie':
        if args.multi:
            args.val_img_dir = ''
        else:
            args.val_img_dir = './train_list/mpie_cross_test_cvpr_full.txt'

    elif args.dataset == 'vox1':

        if args.multi:
            args.val_img_dir = ''
        else:
            args.val_img_dir = ''

    elif args.dataset == 'vox2':
        if args.multi:
            args.val_img_dir = ''
        else:
            args.val_img_dir = ''

    main(args)
