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






if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Eval_SOTA_Model')
    # learning & saving parameters
    # parser.add_argument('-data-place', type=str, default='D:/Josh2/fnm-master_Validate/test_MPIE/FNM-MB16-Lamda_Fea1000_1_0', help='prepared data path to run program')
    # parser.add_argument('-csv-file', type=str, default='../DataList/IJB-A_FOCropped_250_250_84.csv', help='csv file to load image for training')
    parser.add_argument('-data-place', type=str, default='D:/Josh2/dualview-normalization_MultiGraph/test_f/SA_FNM_MB4_Fea3500_051_190_Illum_Sym001_All_w_AllGP_1_0', help='prepared data path to run program')
    parser.add_argument('-csv-file', type=str, default='../DataList/IJB-A_FOCropped_250_250_84.csv', help='csv file to load image for training')
    parser.add_argument('-model-select', type=str, default='Light_CNN_29', help='Model Select')
    parser.add_argument('-cuda', action='store_true', default=True, help='enable the gpu')
    parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 8]')
    # Evaluation options
    # parser.add_argument('-generate-place', type=str, default='D:/Experiment_IJBA_New/Feature/FNM_LightCNN29_Fea15000_FOCropped_250_250_84_epoch7_0', help='prepared data path to run program')
    parser.add_argument('-generate-place', type=str, default='D:/04_FaceEvaluation/Experiment_IJBA_Mean_v02/Feature/SA_FNM_MB4_Fea3500_051_Illum_Sym001_All_w_AllGP_1_0', help='prepared data path to run program')
    parser.add_argument('-Save-Features', action='store_true', default=True, help='enable the gpu')
    parser.add_argument('-Eval-CFP', action='store_true', default=False, help='enable the gpu')

    args = parser.parse_args()


    BACKBONE_DICT = {'IR-50': IR_50(112),
                 'Light_CNN_9': LightCNN_9Layers(),
                 'Light_CNN_29': LightCNN_29Layers(),
                 'Light_CNN_29_v2': LightCNN_29Layers_v2(),
                 'VGGFace2': resnet50_ft(weights_path='Pretrained/VGGFace2/resnet50_ft_dims_2048.pth')
    }
    BACKBONE = BACKBONE_DICT[args.model_select]
    Model = LoadPretrained(BACKBONE, args)

    save_dir = '{}'.format(args.generate_place)
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    if not args.Save_Features and not args.Eval_CFP:
        print('Please select valid option for saving features (args.Save_Features) or evalating on CFP (args.Eval_CFP)')
        print('Loading the default setting (Save_Features)')
        args.Eval_CFP = True


    if args.cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Model = Model.to(device)
    Model.eval()

    # Load augmented data
    transforms = Transform_Select(args)
    transformed_dataset = FaceIdPoseDataset(args.csv_file, args.data_place,
                                            transform=transforms)
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False)  # , num_workers=6)

    Features_List = {'Subject':[], 'Pose':[], 'ImgNum':[], 'Features':[]}
    count = 0
    minibatch_size = args.batch_size
    for i, batch_data in enumerate(dataloader):
        if args.model_select == 'VGGFace2': batch_image = (batch_data[0]*255).to(device)
        else: batch_image = batch_data[0].to(device)

        _ = Model(batch_image)
        try: Model_Feature = Model.feature
        except: Model_Feature = Model.module.feature

        if len(Model_Feature.shape) > 2:
            Model_Feature = Model_Feature.view(Model_Feature.size(0), -1)
        features = (Model_Feature.data).cpu().numpy()
        batchImageName = batch_data[1]

        if args.Save_Features:
            for feature, ImgName in zip(features, batchImageName):
                tmp = ImgName.split('/')
                FeaturePath, SavePath = ConcatPath(save_dir, tmp, '.txt')
                if not os.path.isdir(SavePath): os.makedirs(SavePath)
                with open(FeaturePath, 'w') as f:
                    for fea in feature:
                        text = str(fea)
                        f.write("{}\n".format(text))
            count += minibatch_size
            print("Finish Processing {} images...".format(count))




        if args.Eval_CFP:
            for ImgName, feas in zip(batchImageName, features):
                tmp = ImgName.split('/')
                Pose = PoseSelect(tmp[1])

                Features_List['Subject'].append(int(tmp[0]))
                Features_List['Pose'].append(Pose)
                Features_List['ImgNum'].append(int(tmp[2].split('.')[0]))
                Features_List['Features'].append(feas)

            count += minibatch_size
            print("Finish Processing {} images...".format(count))
        # if count>1000:
        #     break
    if args.Eval_CFP:
        print('Loading the CFP protocol ...')

        PairRoot = './Protocol/Split2Mat'
        ROCFlag = 0
        Show_Flag = 0
        Type = ['FF', 'FP']
        ACCURACY = []
        ACCURACY = np.array(ACCURACY)
        in_fea = []
        ex_fea = []

        Fea = pd.DataFrame.from_dict(Features_List)
        for tp in range(len(Type)):
            SplitList = os.listdir(os.path.join(PairRoot, Type[tp]))
            for s1 in range(0, len(SplitList)):
                Name = ['same.mat', 'diff.mat']
                DISTANCE = []
                LABEL = []
                RESULT_PairIdx = []
                for nn in range(2):
                    DataName = '{}/{}/{}/{}'.format(PairRoot, Type[tp], SplitList[s1], Name[nn])
                    data = sio.loadmat(DataName)  # ,  struct_as_record=False)

                    Path = data['Pair'][0, 0]['Path']

                    for pp in Path:

                        tmp1 = pp[0][0].split('/')
                        tmp2 = pp[1][0].split('/')

                        Pose = PoseSelect(tmp1[4])
                        Fea1 = Fea.Features[Fea[(Fea.Subject == int(tmp1[3])) & (Fea.Pose == Pose) &
                                                (Fea.ImgNum == int(tmp1[5].split('.')[0]))].index.tolist()[0]]

                        Pose = PoseSelect(tmp2[4])
                        Fea2 = Fea.Features[Fea[(Fea.Subject == int(tmp2[3])) & (Fea.Pose == Pose) &
                                                (Fea.ImgNum == int(tmp2[5].split('.')[0]))].index.tolist()[0]]

                        if len(Fea[(Fea.Subject == int(tmp1[3])) & (Fea.Pose == Pose) & (
                                Fea.ImgNum == int(tmp1[5].split('.')[0]))].index.tolist()) > 1 or len(Fea[(
                                                                                                                  Fea.Subject == int(
                                                                                                              tmp2[
                                                                                                                  3])) & (
                                                                                                                  Fea.Pose == Pose) & (
                                                                                                                  Fea.ImgNum == int(
                                                                                                              tmp2[
                                                                                                                  5].split(
                                                                                                                  '.')[
                                                                                                                  0]))].index.tolist()) > 1:
                            exit()

                        # Fea1 = Fea.loc[(Fea['Subject'] == tmp1[3]) & (Fea['Pose'] == tmp1[4]) & (Fea['ImgNum'] == tmp1[5][0:-4] + '.jpg')].values[-1][-1]

                        # distance = cosine_similarity((Fea1/norm(Fea1)).reshape(1, -1), (Fea2/norm(Fea2)).reshape(1, -1))
                        distance = cdist(Fea1.reshape(1, -1) / norm(Fea1.reshape(1, -1)),
                                         Fea2.reshape(1, -1) / norm(Fea2.reshape(1, -1)), 'cosine')
                        DISTANCE.append(distance)
                    LABEL.append(data['Pair'][0, 0]['Label'])
                    # RESULT_PairIdx.append(data['Pair'][0, 0]['Idx'])
                DISTANCE_array = np.array(DISTANCE).reshape(-1, 1)
                # RESULT_PairIdx_array = np.array(RESULT_PairIdx).reshape(-1, 1)
                LABEL_array = np.array(LABEL).reshape(-1, 1)
                Result_TAR = []
                Result_FAR = []
                Result_TAR = np.array(Result_TAR)
                Result_FAR = np.array(Result_FAR)
                Result_BestAcc = 0
                for thresh in range(0, 151):
                    thresh = thresh / 100
                    THRESH = np.ones([len(LABEL_array), 1]) * thresh
                    Intra_predict = [pre_indx for (pre_indx, val) in
                                     enumerate(THRESH[0:350] - DISTANCE_array[0:350]) if
                                     val > 0]
                    extra_predict = [pre_exdx for (pre_exdx, val2) in
                                     enumerate(THRESH[350:700] - DISTANCE_array[350:700])
                                     if val2 < 0]
                    Result_TAR = np.append(Result_TAR, len(Intra_predict) / 350)
                    Result_FAR = np.append(Result_FAR, (350 - len(extra_predict)) / 350)
                    ACC = (len(Intra_predict) + len(extra_predict)) / 700
                    ACC = float('%.4f' % ACC)
                    if ACC > Result_BestAcc:
                        Result_Thresh = thresh
                        Result_BestAcc = ACC
                        Result_Intra_predict = Intra_predict
                        Result_extra_predict = extra_predict
                        Result_Distance = DISTANCE_array
                ACCURACY = np.append(ACCURACY, Result_BestAcc)
                print('>>> Split {} \tBest Accuracy : {} {} \n'.format(s1, Result_BestAcc * 100, '%'))

                in_fea = np.array([])
                ex_fea = np.array([])
                in_fea = np.append(in_fea, DISTANCE_array[0:350])
                ex_fea = np.append(ex_fea, DISTANCE_array[350:700])
            STD = np.std(ACCURACY, ddof=1)
            STD = round(STD, 4)
            MEAN = np.mean(ACCURACY)
            MEAN = float('%.4f' % MEAN)
            print('>>> CFP {} Mean Accuracy : {}{}, Std={}\n'.format(Type[tp], MEAN * 100, '%', STD * 100))
            ACCURACY = np.array([])
            in_fea = np.array([])
            ex_fea = np.array([])

    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)












