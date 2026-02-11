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

from core.puzzle_utils import *
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default='/media/ders/sdc1/FJM/datasets/Generality/P256s256/crop4/', type=str)
###############################################################################
# Inference parameters
###############################################################################
# parser.add_argument('--experiment_name', default='resnet50@seed=0@bs=16@ep=5@nesterov@train@scale=0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--experiment_name', default='resnet50Aff', type=str)
parser.add_argument('--domain', default='train', type=str)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    experiment_name = args.experiment_name
    
    pred_dir = f'/media/ders/sdc1/FJM/115P4mul_ipc_AP_SAM/Generality/experimentsP256s256_crop4/CAM_ipc0.1/predictions/CAMnpy/'  # 存储CAM.npy文件的文件夹
    aff_dir = create_directory('/media/ders/sdc1/FJM/115P4mul_ipc_AP_SAM/Generality/experimentsP256s256_crop4/pseudo_labels/')

    # pred_dir = f'./experimentsI512s512_SFC/CAM/predictions_0.9/CAMnpy/'  # 存储CAM.npy文件的文件夹
    # aff_dir = create_directory('./experimentsI512s512_SFC/pseudo_labels_0.9/label/')
    # aff_dir_vis = create_directory('./experimentsI512s512_SFC/pseudo_labels_0.7/vis/')
    set_seed(args.seed)
    log_func = lambda string='': print(string)

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    # for mIoU
    meta_dic = read_json('./data/VOC_2012.json')
    dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)

    #################################################################################################
    # Convert
    #################################################################################################
    eval_timer = Timer()

    length = len(dataset)
    for step, (ori_image, image_id, _, _) in enumerate(dataset):
        png_path = aff_dir + image_id + '.png'
        # png_path_vis = aff_dir_vis + image_id + '.png'
        if os.path.isfile(png_path):
            continue
        # if os.path.isfile(png_path_vis):
        #     continue

        image = np.asarray(ori_image)
        cam_dict = np.load(pred_dir + image_id + '.npy', allow_pickle=True).item()

        ori_h, ori_w, c = image.shape
        keys = cam_dict['keys']
        cams = cam_dict['hr_cam']
        fg_cam = crf_with_alpha(image, cam_dict)
        fg_cam = np.argmax(fg_cam, axis=0)  # 对应的是keys里边的个数 的通道数，并不是6个通道数
        fg_conf = keys[fg_cam]
        conf = fg_conf.copy()
        # # 定义标签颜色映射
        # def pv2rgb(mask):
        #     h, w = mask.shape[0], mask.shape[1]
        #     mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        #     mask_convert = mask[np.newaxis, :, :]
        #     mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 0, 127]
        #     mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 0, 0]
        #     mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 63, 255]
        #     mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 127, 191]
        #     mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 0, 63]
        #     mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 127, 63]
        #     mask_rgb[np.all(mask_convert == 6, axis=0)] = [0, 63, 191]
        #     mask_rgb[np.all(mask_convert == 7, axis=0)] = [0, 63, 0]
        #     mask_rgb[np.all(mask_convert == 8, axis=0)] = [0, 191, 127]
        #     mask_rgb[np.all(mask_convert == 9, axis=0)] = [0, 127, 255]
        #     mask_rgb[np.all(mask_convert == 10, axis=0)] = [0, 63, 127]
        #     mask_rgb[np.all(mask_convert == 11, axis=0)] = [0, 127, 127]
        #     mask_rgb[np.all(mask_convert == 12, axis=0)] = [0, 63, 63]
        #     mask_rgb[np.all(mask_convert == 13, axis=0)] = [0, 100, 155]
        #     mask_rgb[np.all(mask_convert == 14, axis=0)] = [0, 0, 255]
        #     mask_rgb[np.all(mask_convert == 15, axis=0)] = [0, 0, 191]
        #     return mask_rgb
        # def pv2rgb(mask):
        #     h, w = mask.shape[0], mask.shape[1]
        #     mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        #     mask_convert = mask[np.newaxis, :, :]
        #     mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
        #     mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
        #     mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 255]
        #     mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 255, 255]
        #     mask_rgb[np.all(mask_convert == 4, axis=0)] = [255, 204, 0]
        #     mask_rgb[np.all(mask_convert == 5, axis=0)] = [255, 0, 0]
        #     return mask_rgb
        # 3.whdld可视化标签图片
        # def pv2rgb(mask):
        #     h, w = mask.shape[0], mask.shape[1]
        #     mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        #     mask_convert = mask[np.newaxis, :, :]
        #     mask_rgb[np.all(mask_convert == 3, axis=0)] = [255, 255, 0]
        #     mask_rgb[np.all(mask_convert == 0, axis=0)] = [128, 128, 128]
        #     mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
        #     mask_rgb[np.all(mask_convert == 2, axis=0)] = [192, 192, 0]
        #     mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 255, 0]
        #     mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
        #     return mask_rgb


        # mask = pv2rgb(conf)
        # plt.imshow(mask)
        # plt.show()
        imageio.imwrite(png_path, conf.astype(np.uint8))
        # imageio.imwrite(png_path_vis, mask.astype(np.uint8))
        
        sys.stdout.write('\r# Convert [{}/{}] = {:.2f}%, ({}, {})'.format(step + 1, length, (step + 1) / length * 100, (ori_h, ori_w), conf.shape))
        sys.stdout.flush()
    print()
    