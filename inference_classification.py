# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--data_dir', default='/media/ders/sdc1/FJM/datasets/Generality/P256s256/crop4/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str)
###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--tag', default='CAM', type=str)
parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
# parser.add_argument('--scales', default='1.0', type=str)
if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    experiment_name = args.tag

    if 'train' in args.domain:
        experiment_name += '@train'
    else:
        experiment_name += '@val'

    experiment_name += '@scale=%s'%args.scales
    
    pred_dir = create_directory('/media/ders/sdc1/FJM/115P4mul_ipc_AP_SAM/Generality/experimentsP256s256_crop4/CAM_ipc0.1/predictions/CAMnpy/')
    # CAM_Image_Vis_dir = create_directory('./experimentsI512s512_SFC/CAM/predictions_0.9/CAMVis/')
    model_path = '/media/ders/sdc1/FJM/115P4mul_ipc_AP_SAM/Generality/experimentsP256s256_crop4/CAM_ipc0.1/models/' + f'{args.tag}.pth'

    # pred_dir = create_directory(f'./experimentsI512s512_SFC/CAM/predictions_0.9/CAMnpy/')
    # CAM_Image_Vis_dir = create_directory('./experimentsI512s512_SFC/CAM/predictions_0.9/CAMVis/')
    # # model_path = '/media/ders/sdd1/FJM/823P3mul_ipc_AP_CAM/experimentsV128s128/CAM_ipc0.4_pcm1th0.6/models/' + f'{args.tag}.pth'
    # model_path = './experimentsI512s512_SFC/CAM/models/' + f'{args.tag}.pth'

    set_seed(args.seed)
    log_func = lambda string='': print(string)

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    # for mIoU
    meta_dic = read_json('./data/VOC_2012.json')
    dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)
    
    ###################################################################################
    # Network
    ###################################################################################
    model = Classifier(args.architecture, meta_dic['classes'], mode=args.mode)
    model = model.cuda()
    model.eval()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    load_model(model, model_path, parallel=the_number_of_gpu > 1)
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]  # 存储四种缩放尺度的图片
    
    model.eval()
    eval_timer.tik()
    def get_cam(ori_image, scale):
        # preprocessing
        image = copy.deepcopy(ori_image)
        image = image.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.CUBIC)

        image = normalize_fn(image)
        image = image.transpose((2, 0, 1))

        image = torch.from_numpy(image)
        flipped_image = image.flip(-1)  # 原图像进行翻转，提高程序的性能

        images = torch.stack([image, flipped_image])
        images = images.cuda()

        # inferenece
        _, features = model(images, with_cam=True)  # backbone最后一层2 5 8 8
        # features, _ = model(images, with_cam=True)  # backbone最后一层2 5 8 8

        # postprocessing
        cams = F.relu(features)  # 像素值处理至大于0
        cams = cams[0] + cams[1].flip(-1)  # 变为5 8 8 对于每一个输入到网络中的图片x[0]，我们将其翻转后x[1]与他原本的图片一起输入网络处理，然后融合

        return cams

    with torch.no_grad():
        length = len(dataset)
        for step, (ori_image, image_id, label, gt_mask) in enumerate(dataset):
            ori_w, ori_h = ori_image.size  # 当前图片大小

            npy_path = pred_dir + image_id + '.npy'
            # CAM_Image_Vis_dir_path = CAM_Image_Vis_dir + image_id + '.png'
            if os.path.isfile(npy_path):
                continue
            strided_size = get_strided_size((ori_h, ori_w), 4)  # 下采样
            strided_up_size = get_strided_up_size((ori_h, ori_w), 16)  # 上采样

            cams_list = [get_cam(ori_image, scale) for scale in scales]  # 将同一张图片缩放成四张图片分别送入网络然后进行预测

            strided_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_size)[0] for cams in cams_list]  # 预测图进行上采样
            strided_cams = torch.sum(torch.stack(strided_cams_list), dim=0)  # 先拼接再求和

            hr_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_up_size)[0] for cams in cams_list]
            hr_cams = torch.sum(torch.stack(hr_cams_list), dim=0)[:, :ori_h, :ori_w]  # 将四个尺度的和加在一起，构成总体。1张CAM图；6 256 256每一个结果恢复到原图大小

            keys = torch.nonzero(torch.from_numpy(label))[:, 0]  # 找到标签中标为1的类别，也就是当前图片中存在的类别

            strided_cams = strided_cams[keys]  # 将存在的类别保持，不存在的类别的值变为0
            strided_cams /= F.adaptive_max_pool2d(strided_cams, (1, 1)) + 1e-5
            hr_cams = hr_cams[keys]  # 从hr_cams张量中选取特定的元素或子张量，具体选取哪些元素取决于keys变量的取值
            hr_cams /= F.adaptive_max_pool2d(hr_cams, (1, 1)) + 1e-5  # 4 256 256 tensor形式
            # hr_cams = hr_cams[keys]
            # hr_cams /= F.adaptive_max_pool2d(hr_cams, (1, 1)) + 1e-5  # 4 256 256 tensor形式
            # hr_camsnpy = hr_cams
            ############################## 热力图可视化####################
            # for idx in range(0, hr_cams.size(0)):
            #     final_cam = hr_cams[idx, :, :]
            #     final_cam = final_cam.cpu().numpy()
            #     # img = np.float32(img) / 255.
            #     heatmap = cv2.applyColorMap(np.uint8(255 * final_cam), cv2.COLORMAP_JET)
            #     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            #     ori_image = np.array(ori_image)
            #     cam = heatmap * 0.5 + ori_image * 0.3
            #     cam = cam / np.max(cam)
            #     cam = np.uint8(255 * cam)
            #     CAM_Image_Vis_dir_path = CAM_Image_Vis_dir + image_id + '_' + str(idx) + '.png'
            #     imageio.imwrite(CAM_Image_Vis_dir_path, cam)  # 存储热力图+image
            ############################# 热力图可视化结束####################
            # plt.imshow(cam)
            # plt.show()
            keys = np.pad(keys, (0, 0), mode='constant')
            np.save(npy_path, {"keys": keys, "cam": strided_cams.cpu(), "hr_cam": hr_cams.cpu().numpy()})  # 存的是3 256 256的数组
            sys.stdout.write('\r# Make CAM [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100, (ori_h, ori_w), hr_cams.size()))
            sys.stdout.flush()
        print()
    
    if args.domain == 'train_aug':
        args.domain = 'train'
    
    print("python3 evaluate.py --experiment_name {} --domain {}".format(experiment_name, args.domain))