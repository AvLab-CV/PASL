import cv2 
import argparse
import face_alignment
import torch
import argparse
import cv2
import numpy as np
import argparse
import cv2
import numpy as np
import torch
from Instant import Solver, toTensor, get_arcface
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

def crop_face_with_landmarks(frame, landmarks):
    x_min = int(np.min(landmarks[:, 0]))
    x_max = int(np.max(landmarks[:, 0]))
    y_min = int(np.min(landmarks[:, 1]))
    y_max = int(np.max(landmarks[:, 1]))
    margin = 20  # 增加一些邊距
    x_min = max(0, x_min - margin)
    x_max = min(frame.shape[1], x_max + margin)
    y_min = max(0, y_min - margin)
    y_max = min(frame.shape[0], y_max + margin)
    return frame[y_min:y_max, x_min:x_max]

def add_text(img, text, pos):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = pos[0] + (img.shape[1] - text_size[0]) // 2
    text_y = pos[1] + text_size[1] + 10
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
      
def main_face_video(video):
    solver = Solver(args)
    print("Loading the FAN Model......")
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device='cuda')

    cap = cv2.VideoCapture(video)
    
    img_size = 256
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor()
    ])

    # 讀取來源圖片
    source_image = cv2.imread('./examples/musk.png')
    source_image = cv2.resize(source_image, (512, 512))

    # 讀取範例圖片
    img_crop = cv2.imread('./examples/96.jpg')
    img_crop_2 = cv2.imread('./examples/287.jpg')
    img_crop_3 = cv2.imread('./examples/1466_01_03_0018.jpg')
    img_crop_4 = cv2.imread('./examples/1465_01_03_0055.jpg')

    img_crop = cv2.resize(img_crop, (256, 256))
    img_crop_2 = cv2.resize(img_crop_2, (256, 256))
    img_crop_3 = cv2.resize(img_crop_3, (256, 256))
    img_crop_4 = cv2.resize(img_crop_4, (256, 256))

    deca = DECA(config=deca_cfg, device='cuda')
    face_detector = detectors.FAN()

    source = toTensor(img_crop)
    source_2 = toTensor(img_crop_2)
    source_3 = toTensor(img_crop_3)
    source_4 = toTensor(img_crop_4)

    source_all = torch.zeros((4, 3, 256, 256), dtype=torch.float32)
    source_all[0, :, :, :] = source
    source_all[1, :, :, :] = source_2
    source_all[2, :, :, :] = source_3
    source_all[3, :, :, :] = source_4
    source_all = source_all.type(torch.HalfTensor).cuda()

    source_style_code = solver.extract(source_all)
    idx = 0
    freq = 2

    # 定義影片編碼格式和輸出影片對象
    output_video = './output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 24  # 可以根據需要調整
    out = cv2.VideoWriter(output_video, fourcc, fps, (1536, 512))  

    # cv2.namedWindow('Combined Image', cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty('Combined Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    
    
    while cap.isOpened():
        idx += 1
        ret = cap.grab()
        if idx % freq == 0:
            ret, frame_bgr = cap.retrieve()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # 正確的 RGB 轉換
            
            preds = fa.get_landmarks(frame_rgb)
            if preds is not None:
                landmarks = preds[0]  # 假設只檢測到一張臉
                frame_bgr_cropped = crop_face_with_landmarks(frame_bgr, landmarks)

                try:
                    img_crop2 = get_arcface(frame_rgb, preds)
                    depth_image_list, lm_image_list = get_shape(deca, face_detector, np.array(img_crop2))
                except Exception as e:
                    continue

                lm_shape = torch.zeros((4, 3, 256, 256), dtype=torch.float32)
                pncc_map = torch.zeros((4, 3, 256, 256), dtype=torch.float32)

                for i in range(4):
                    lm_shape[i] = transform(lm_image_list[i])
                    pncc_map[i] = transform(depth_image_list[i])

                img_fake, img_fake_2, img_fake_3, img_fake_4 = solver.sample(source_style_code, pncc_map, lm_shape)

                all_result = np.zeros((512, 512, 3), np.uint8)
                img_fake = cv2.resize(img_fake, (256, 256))
                img_fake_2 = cv2.resize(img_fake_2, (256, 256))
                img_fake_3 = cv2.resize(img_fake_3, (256, 256))
                img_fake_4 = cv2.resize(img_fake_4, (256, 256))

                all_result[:256, :256, :] = cv2.cvtColor(img_fake, cv2.COLOR_RGB2BGR)
                all_result[:256, 256:512, :] = cv2.cvtColor(img_fake_2, cv2.COLOR_RGB2BGR)
                all_result[256:512, :256, :] = cv2.cvtColor(img_fake_3, cv2.COLOR_RGB2BGR)
                all_result[256:512, 256:512, :] = cv2.cvtColor(img_fake_4, cv2.COLOR_RGB2BGR)

                # 將來源圖片和處理後的影像垂直合併
                #right_result = np.vstack((source_image, all_result))
                

                # 將影片幀和右側結果水平合併
                frame_resized = cv2.resize(frame_bgr_cropped, (512, 512))  # 調整為適當的大小以匹配右側結果的高度
                source_image_resized = cv2.resize(source_image, (512, 512))
                #combined_result = np.vstack((frame_resized, source_image, all_result))
                #combined_result = np.hstack((frame_resized, right_result))  # 顯示 BGR 格式
                combined_result = np.hstack((frame_resized, source_image_resized, all_result))
        
                cv2.imshow('Combined Image', combined_result)

                out.write(combined_result)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "examples/jennacut.mp4"
    main_face_video(video_path)
