import gradio as gr
import cv2 
import argparse
import face_alignment
import torch
from Instant import Solver, toTensor, get_arcface
from typing import Iterable 
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import time
import threading
import numpy as np

from PIL import Image 
from decalib.deca import DECA
from decalib.datasets import datasets_demo 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.datasets import detectors
from torchvision import transforms

#----------------------------------
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
parser.add_argument('--self_att', action='store_true', default=True,
                    help='use self-attention')
parser.add_argument('--w_hpf', type=float, default=1,
                    help='weight for high-pass filtering')
parser.add_argument('--resume_iter', type=int, default=204000,
                    help='Iterations to resume training/testing')
parser.add_argument('--val_batch_size', type=int, default=32,
                    help='Batch size for validation')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate for D, E and G')
parser.add_argument('--checkpoint_dir', type=str, default='./experiment/demo',
                    help='Directory for saving network checkpoints')
args = parser.parse_args()
#-----------------------------------------------------------------------------------
# multi thread
class ipcamCapture:
    def __init__(self, url):
        self.Frame = []
        self.status = False
        self.isstop = False
		
	# 攝影機連接。
        self.capture = cv2.VideoCapture(url)

    def start(self):
	# 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        print('ipcam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
	# 記得要設計停止無限迴圈的開關。
        self.isstop = True
        print('ipcam stopped!')
   
    def getframe(self):
	# 當有需要影像時，再回傳最新的影像。
        return self.Frame.copy()
        
    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()
        
        self.capture.release()
#-----------------------------------------------------------------------------------

def fuse_shape(deca, face_detector, src_codedict, ref_img):
    testdata = datasets_demo.TestData(face_detector, [ref_img], iscrop=True, face_detector='fan', sample_step=10)
    device = 'cuda'
    i=0
    depth_image_list = []
    lm_image_list = []
    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = 'pytorch3d'
    deca_cfg.model.extract_tex = True
    img = testdata[i]['image'].to(device)[None,...]
    with torch.no_grad():
        ref_codedict = deca.encode(img)
    for i in range(len(src_codedict)):
        with torch.no_grad():
            codedict2 = ref_codedict
            codedict1 = src_codedict[i]
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
            tform = testdata[0]['tform'][None, ...]
            tform = torch.inverse(tform).transpose(1,2).to(device)
            original_image = testdata[0]['original_image'][None, ...].to(device)
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
            depth_image_list.append(depth_image)
            lm_image_list.append(lm_image)
    return depth_image_list, lm_image_list

def get_shape(deca, face_detector, ref_img):
    testdata = datasets_demo.TestData(face_detector, [ref_img], iscrop=True, face_detector='fan', sample_step=10)
    device = 'cuda'
    i=0
    depth_image_list = []
    lm_image_list = []
    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = 'pytorch3d'
    deca_cfg.model.extract_tex = True
    img = testdata[i]['image'].to(device)[None,...]
    with torch.no_grad():
        ref_codedict = deca.encode(img)
        temp = ref_codedict
        tform = testdata[0]['tform'][None, ...]
        tform = torch.inverse(tform).transpose(1,2).to(device)
        original_image = testdata[0]['original_image'][None, ...].to(device)
        orig_opdict, orig_visdict = deca.decode(temp, render_orig=True, original_image=original_image, tform=tform)    
        # orig_visdict['inputs'] = original_image
        # cv2.imwrite('1.png', cv2.resize(util.tensor2image(orig_visdict['landmarks2d'][0]),(256,256)))
        lm_image = cv2.resize(util.tensor2image(orig_visdict['landmarks2d'][0]),(256,256))
        depth_image = deca.render.render_depth(orig_opdict['trans_verts']).repeat(1,3,1,1)[0]

        depth_image = depth_image.detach().cpu().numpy()
        depth_image = depth_image*255.
        depth_image = np.maximum(np.minimum(depth_image, 255), 0)
        depth_image = depth_image.transpose(1,2,0)[:,:,[2,1,0]]
        depth_image = Image.fromarray(np.uint8(depth_image))
        lm_image = Image.fromarray(lm_image)
        for k in range(4):
            depth_image_list.append(depth_image)
            lm_image_list.append(lm_image)
    return depth_image_list, lm_image_list


def get_codedict(deca, face_detector, img_list):
    testdata = datasets_demo.TestData(face_detector, img_list, iscrop=True, face_detector='fan', sample_step=10)
    device = 'cuda'
    i=0
    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = 'pytorch3d'
    deca_cfg.model.extract_tex = True
    # src_name = testdata[i]['imagename']
    codedict_list = []
    for i in range(len(testdata)):
        img = testdata[i]['image'].to(device)[None,...]
        with torch.no_grad():
            codedict = deca.encode(img)
            codedict_list.append(codedict)
    return codedict_list

def main_face(kkk):
    solver = Solver(args)
    # cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    print("Loading the FAN Model......")
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device='cuda')
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'
        # os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF']= '0'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

    #     face_boxes = FaceBoxes_ONNX()
    #     tddfa = TDDFA_ONNX(**cfg)
    # else:
    #     gpu_mode = args.mode == 'gpu'
    #     tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    #     face_boxes = FaceBoxes()

    # Given a camera
    # before run this line, make sure you have installed `imageio-ffmpeg`
    # reader = imageio.get_reader("<video0>") #090909090


    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FPS, 4)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    # n_pre, n_next = args.n_pre, args.n_next
    # n = n_pre + n_next + 1
    # queue_ver = deque()
    # queue_frame = deque()

    # # run
    # dense_flag = args.opt in ('2d_dense', '3d', 'pncc')
    # pre_ver = None
    img_size = 256
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor()
    ])
    #input img
    img_crop = cv2.imread('./examples/96.jpg')
    img_crop_2 = cv2.imread('./examples/287.jpg')
    img_crop_3 = cv2.imread('./examples/1465_01_03_0055.jpg')
    img_crop_4 = cv2.imread('./examples/1466_01_03_0018.jpg')

    img_crop = cv2.resize(img_crop, (256, 256))
    img_crop_2 = cv2.resize(img_crop_2, (256, 256))
    img_crop_3 = cv2.resize(img_crop_3, (256, 256))
    img_crop_4 = cv2.resize(img_crop_4, (256, 256))

    # src_codedict = get_codedict(deca, face_detector, [img_crop, img_crop_2, img_crop_3 ,img_crop_4])

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
    # i = 0
    # idx = 0
    # freq = 2

    cap = ipcamCapture(0)
    cap.start()
    time.sleep(1)
    deca = DECA(config = deca_cfg, device='cuda')
    face_detector = detectors.FAN()
    while True:
    # for i, frame in tqdm(enumerate(video)):
        # ret, frame = cap.read()
        frame = cap.getframe()
        frame_bgr = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        preds = fa.get_landmarks(frame_bgr)
        try:
            # img_crop2, img_shape, shape = get_arcface(frame_bgr, preds)
            # depth_image_list, lm_image_list = fuse_shape(deca, face_detector, src_codedict, np.array(img_crop2))
            img_crop2 = get_arcface(frame_bgr, preds)
            depth_image_list, lm_image_list = get_shape(deca, face_detector, np.array(img_crop2))
        except:
            img_crop2 = cv2.imread('./examples/smiling.png')
            all = cv2.imread('./examples/smiling.png')
            yield img_crop2,all
            continue

        # if i == 0:
        #     # the first frame, detect face, here we only use the first face, you can change depending on your need

        #     boxes = face_boxes(img_crop2)
        #     if boxes==[]:continue
        #     boxes = [boxes[0]]
        #     param_lst, roi_box_lst = tddfa(img_crop2, boxes)
        #     ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

        #     # refine
        #     param_lst, roi_box_lst = tddfa(img_crop2, [ver], crop_policy='landmark')
        #     ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            
        #     # padding queue
        #     for _ in range(n_pre):
        #         queue_ver.append(ver.copy())
        #     queue_ver.append(ver.copy())

        #     for _ in range(n_pre):
        #         queue_frame.append(img_crop2.copy())
        #     queue_frame.append(img_crop2.copy())
        # else:
        #     try:
        #         param_lst, roi_box_lst = tddfa(img_crop2, [pre_ver], crop_policy='landmark')
        #     except:
        #         continue

        #     roi_box = roi_box_lst[0]
        #     # todo: add confidence threshold to judge the tracking is failed
        #     if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
        #         boxes = face_boxes(img_crop2)
        #         boxes = [boxes[0]]
        #         param_lst, roi_box_lst = tddfa(img_crop2, boxes)

        #     ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

        #     queue_ver.append(ver.copy())
        #     queue_frame.append(img_crop2.copy())

        # pre_ver = ver  # for tracking

        # smoothing: enqueue and dequeue ops
        # if len(queue_ver) >= n:
        #     ver_ave = np.mean(queue_ver, axis=0)

        #     if args.opt == '2d_sparse':
        #         img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)  # since we use padding
        #     elif args.opt == '2d_dense':
        #         img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
        #     elif args.opt == '3d':
        #         img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
        #     elif args.opt == 'pncc':
        #         img_draw = pncc(queue_frame[n_pre], [ver_ave], tddfa.tri, show_flag=None, wfp=None, with_bg_flag=False)
        #     else:
        #         raise ValueError(f'Unknown opt {args.opt}')
            
        # img_shape = toTensor(img_shape)
        lm_shape = np.zeros((4, 3, 256, 256))
        lm_shape = torch.from_numpy(lm_shape)
        lm_shape= lm_shape.float()

        lm_shape[0,:,:,:] = transform(lm_image_list[0])
        lm_shape[1,:,:,:] = transform(lm_image_list[1])
        lm_shape[2,:,:,:] = transform(lm_image_list[2])
        lm_shape[3,:,:,:] = transform(lm_image_list[3])
        # img_draw = toTensor(img_draw)

        pncc_map = np.zeros((4, 3, 256, 256))
        pncc_map = torch.from_numpy(pncc_map)
        pncc_map= pncc_map.float()

        pncc_map[0,:,:,:] = transform(depth_image_list[0])
        pncc_map[1,:,:,:] = transform(depth_image_list[1])
        pncc_map[2,:,:,:] = transform(depth_image_list[2])
        pncc_map[3,:,:,:] = transform(depth_image_list[3])
        img_fake, img_fake_2, img_fake_3, img_fake_4 = solver.sample(source_style_code, pncc_map, lm_shape)
        all = np.zeros((512, 512, 3), np.uint8)

        img_fake = cv2.resize(img_fake, (256,256))
        img_fake_2 = cv2.resize(img_fake_2, (256,256))
        img_fake_3 = cv2.resize(img_fake_3, (256,256))
        img_fake_4 = cv2.resize(img_fake_4, (256,256))

        all[:256, :256, :] = img_fake
        all[:256, 256:512, :] = img_fake_2
        all[256:512, :256, :] = img_fake_3
        all[256:512, 256:512, :] = img_fake_4
        
        # queue_ver.popleft()
        # queue_frame.popleft()
        yield img_crop2,all #modify

#-------------------------------------------------------------------------------------
def main_face_video(video):
    solver = Solver(args)
    # cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    print("Loading the FAN Model......")
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device='cuda')
    # if args.onnx:
    #     import os
    #     os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    #     os.environ['OMP_NUM_THREADS'] = '4'

    #     from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
    #     from TDDFA_ONNX import TDDFA_ONNX

    #     face_boxes = FaceBoxes_ONNX()
    #     tddfa = TDDFA_ONNX(**cfg)
    # else:
    #     gpu_mode = args.mode == 'gpu'
    #     tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    #     face_boxes = FaceBoxes()

    
    cap = cv2.VideoCapture(video)
        
    # flag = 0
    # n_pre, n_next = args.n_pre, args.n_next
    # n = n_pre + n_next + 1
    # queue_ver = deque()
    # queue_frame = deque()

    # run
    # dense_flag = args.opt in ('2d_dense', '3d', 'pncc')
    # pre_ver = None
    img_size = 256
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor()
    ])

    #input img
    img_crop = cv2.imread('./examples/96.jpg')
    img_crop_2 = cv2.imread('./examples/287.jpg')
    img_crop_3 = cv2.imread('./examples/1465_01_03_0055.jpg')
    img_crop_4 = cv2.imread('./examples/1466_01_03_0018.jpg')
    # img_crop = cv2.imread('./examples/0199_01_03_0010.jpg')
    # img_crop_2 = cv2.imread('./examples/0262_03_02_0020.jpg')
    # img_crop_3 = cv2.imread('./examples/0548_01_04_0001.jpg')
    # img_crop_4 = cv2.imread('./examples/0367_01_03_0004.jpg')


    img_crop = cv2.resize(img_crop, (256, 256))
    img_crop_2 = cv2.resize(img_crop_2, (256, 256))
    img_crop_3 = cv2.resize(img_crop_3, (256, 256))
    img_crop_4 = cv2.resize(img_crop_4, (256, 256))

    deca = DECA(config = deca_cfg, device='cuda')
    face_detector = detectors.FAN()
    # src_codedict = get_codedict(deca, face_detector, [np.array(img_crop), np.array(img_crop_2), np.array(img_crop_3), np.array(img_crop_4)])

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
    idx = 0
    freq = 2
    
    while (cap.isOpened()):
    # for i, frame in tqdm(enumerate(reader)):
        idx += 1
        ret = cap.grab()
        if idx % freq ==0:
            ret, frame_bgr = cap.retrieve()
            frame_bgr = cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
            preds = fa.get_landmarks(frame_bgr)
            try:
                # img_crop2, img_shape, shape = get_arcface(frame_bgr, preds)
                # depth_image_list, lm_image_list = fuse_shape(deca, face_detector, src_codedict, np.array(img_crop2))
                img_crop2 = get_arcface(frame_bgr, preds)
                depth_image_list, lm_image_list = get_shape(deca, face_detector, np.array(img_crop2))
            except:
                img_crop2 = cv2.imread('./examples/smiling.png')
                all = cv2.imread('./examples/smiling.png')
                yield img_crop2,all
                continue

            # if i == 0:
            #     # the first frame, detect face, here we only use the first face, you can change depending on your need

            #     boxes = face_boxes(img_crop2)
            #     if boxes==[]:continue
            #     boxes = [boxes[0]]
            #     param_lst, roi_box_lst = tddfa(img_crop2, boxes)
            #     ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            #     # refine
            #     param_lst, roi_box_lst = tddfa(img_crop2, [ver], crop_policy='landmark')
            #     ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
                
            #     # padding queue
            #     for _ in range(n_pre):
            #         queue_ver.append(ver.copy())
            #     queue_ver.append(ver.copy())

            #     for _ in range(n_pre):
            #         queue_frame.append(img_crop2.copy())
            #     queue_frame.append(img_crop2.copy())
            # else:
            #     try:
            #         param_lst, roi_box_lst = tddfa(img_crop2, [pre_ver], crop_policy='landmark')
            #     except:
            #         continue

            #     roi_box = roi_box_lst[0]
            #     # todo: add confidence threshold to judge the tracking is failed
            #     if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
            #         boxes = face_boxes(img_crop2)
            #         boxes = [boxes[0]]
            #         param_lst, roi_box_lst = tddfa(img_crop2, boxes)

            #     ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            #     queue_ver.append(ver.copy())
            #     queue_frame.append(img_crop2.copy())

            # pre_ver = ver  # for tracking

            # smoothing: enqueue and dequeue ops
            # if len(queue_ver) >= n:
            #     ver_ave = np.mean(queue_ver, axis=0)

            #     if args.opt == '2d_sparse':
            #         img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)  # since we use padding
            #     elif args.opt == '2d_dense':
            #         img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
            #     elif args.opt == '3d':
            #         img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
            #     elif args.opt == 'pncc':
            #         img_draw = pncc(queue_frame[n_pre], [ver_ave], tddfa.tri, show_flag=None, wfp=None, with_bg_flag=False)
            #     else:
            #         raise ValueError(f'Unknown opt {args.opt}')
                
            # img_shape = toTensor(img_shape)
            lm_shape = np.zeros((4, 3, 256, 256))
            lm_shape = torch.from_numpy(lm_shape)
            lm_shape= lm_shape.float()

            lm_shape[0,:,:,:] = transform(lm_image_list[0])
            lm_shape[1,:,:,:] = transform(lm_image_list[1])
            lm_shape[2,:,:,:] = transform(lm_image_list[2])
            lm_shape[3,:,:,:] = transform(lm_image_list[3])
            # img_draw = toTensor(img_draw)

            pncc_map = np.zeros((4, 3, 256, 256))
            pncc_map = torch.from_numpy(pncc_map)
            pncc_map= pncc_map.float()

            pncc_map[0,:,:,:] = transform(depth_image_list[0])
            pncc_map[1,:,:,:] = transform(depth_image_list[1])
            pncc_map[2,:,:,:] = transform(depth_image_list[2])
            pncc_map[3,:,:,:] = transform(depth_image_list[3])
            img_fake, img_fake_2, img_fake_3, img_fake_4 = solver.sample(source_style_code, pncc_map, lm_shape)
            all = np.zeros((512, 512, 3), np.uint8)

            img_fake = cv2.resize(img_fake, (256,256))
            img_fake_2 = cv2.resize(img_fake_2, (256,256))
            img_fake_3 = cv2.resize(img_fake_3, (256,256))
            img_fake_4 = cv2.resize(img_fake_4, (256,256))

            all[:256, :256, :] = img_fake
            all[:256, 256:512, :] = img_fake_2
            all[256:512, :256, :] = img_fake_3
            all[256:512, 256:512, :] = img_fake_4
            
            # queue_ver.popleft()
            # queue_frame.popleft()
            yield img_crop2,all #modify

#---------------------------------------------------------------------------------------------------------
from typing import Union, Iterable

class Seafoam(Base):
    def __init__(
        self,
        *,
        primary_hue: Union[colors.Color, str] = colors.orange,
        secondary_hue: Union[colors.Color, str] = colors.fuchsia,
        neutral_hue: Union[colors.Color, str] = colors.blue,
        spacing_size: Union[sizes.Size, str] = sizes.spacing_md,
        radius_size: Union[sizes.Size, str] = sizes.radius_md,
        text_size: Union[sizes.Size, str] = sizes.text_lg,
        font: Union[fonts.Font, str, Iterable[Union[fonts.Font, str]]] = (
            fonts.GoogleFont("Quicksand"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: Union[fonts.Font, str, Iterable[Union[fonts.Font, str]]] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="url('https://fastly.picsum.photos/id/213/4928/3264.jpg?hmac=OC0yPL-iiM1YgVjpAjbMf51MjnR6cycgmn1TSLJhDZ0') no-repeat center/cover",
            body_background_fill_dark="repeating-linear-gradient(45deg, *primary_800, *primary_800 10px, *primary_900 10px, *primary_900 20px)",
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
            button_primary_text_color="white",
            # button_secondary_text_color="blue",
            button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="6px",
            block_shadow="*shadow_drop_lg",
            button_shadow="*shadow_drop_lg",
            button_large_padding="32px",
        )

css='''
#image_upload{min-height:400px}
#image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 400px}
'''


seafoam = Seafoam()

with gr.Blocks(theme=seafoam, css=css) as demo:
    with gr.Row():
        gr.Markdown("""# Webcam Input""")
        gr.Markdown("# Source Image")
        gr.Markdown("# Tranformed Faces")
    with gr.Row():
        # crop_frames = gr.Image(label="webcam input")
        crop_frames = gr.Image().style(height=480,width=480)
        source_image = gr.Image('./examples/musk.png').style(height=480,width=480)
        output_video = gr.Image(label="output").style(height=480,width=480)
    # with gr.Row():
    #     gr.Markdown("""# Source Image""")
    # with gr.Row():
    #     gr.Markdown("""![image](file/examples/352.jpg)""")
    #     gr.Markdown("""![image](file/examples/287.jpg)""")
    #     gr.Markdown("""![image](file/examples/96.jpg)""")
    #     gr.Markdown("""![image](file/examples/514.jpg)""")
    with gr.Row():
        with gr.Column():
            gr.Markdown("""# MP4 Input""")
            input_video = gr.Video(label="video input",value='examples/Tom.mp4')
        with gr.Column():
            gr.Markdown("""# Buttons""")
            with gr.Row():
                process_video_btn = gr.Button("Run (webcam)")
                stop = gr.Button(value="Stop", variant="primary")
                done_btn = gr.Button("Run")
            with gr.Column():
                gr.Markdown("""# Examples""")
                with gr.Row():
                    examples = gr.Examples(["examples/Tom.mp4"], inputs=input_video, label=' ')
                    examples = gr.Examples(["examples/wick.mp4"], inputs=input_video, label=' ')
                    examples = gr.Examples(["examples/jennacut.mp4"], inputs=input_video, label=' ')

    cam_event = process_video_btn.click(main_face, 
                            input_video,
                            [crop_frames,output_video])
    
    ex_event = done_btn.click(main_face_video,
                   input_video,
                   [crop_frames, output_video])
    
    stop.click(fn=None, inputs=None, outputs=None, cancels=[ex_event,cam_event])
    output_video.edit(None, None, None, cancels=[cam_event, ex_event])
    # output_video.edit()
    # output_video.play(None, None, None, cancels=[cam_event, ex_event])

demo.queue()
demo.launch(debug=True)
