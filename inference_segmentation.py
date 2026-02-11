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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default='/media/ders/sdc1/FJM/datasets/Generality/P256s256/crop4/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='DeepLabv3', type=str)
parser.add_argument('--backbone', default='resnet50', type=str)
parser.add_argument('--mode', default='fix', type=str)
parser.add_argument('--use_gn', default=True, type=str2bool)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--tag', default='model', type=str)

parser.add_argument('--domain', default='test', type=str)

parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--iteration', default=0, type=int)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    # model_dir = create_directory('./experimentsI512s512_SFC/segmentation/models/')
    model_dir = create_directory(
        '/media/ders/sdc1/FJM/115P4mul_ipc_AP_SAM/Generality/experimentsP256s256_crop4/segmentation/models/')
    model_path = model_dir + f'{args.tag}.pth'

    if 'train' in args.domain:
        args.tag += '@train'
    else:
        args.tag += '@' + args.domain
    
    args.tag += '@scale=%s'%args.scales
    args.tag += '@iteration=%d'%args.iteration

    pred_dir = create_directory('/media/ders/sdc1/FJM/115P4mul_ipc_AP_SAM/Generality/experimentsP256s256_crop4/segmentation/predictions/')
    # pred_dir = create_directory('./experimentsI512s512_SFC/segmentation/predictions/pred_label/')
    # pred_vis_dir = create_directory('./experimentsI512s512_SFC/segmentation/predictions/pred_label_vis/')
    
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
    dataset = VOC_Dataset_For_Evaluation(args.data_dir, args.domain)
    
    ###################################################################################
    # Network
    ###################################################################################
    if args.architecture == 'DeepLabv3':
        model = DeepLabv3_Plus(args.backbone, num_classes=meta_dic['classes'], mode=args.mode, use_group_norm=args.use_gn)
    elif args.architecture == 'Seg_Model':
        model = Seg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)
    elif args.architecture == 'CSeg_Model':
        model = CSeg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)
    
    model = model.cuda()
    model.eval()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    load_model(model, model_path, parallel=False)
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]
    
    model.eval()
    eval_timer.tik()

    def inference(images, image_size):
        images = images.cuda()
        
        logits = model(images)
        logits = resize_for_tensors(logits, image_size)
        
        logits = logits[0] + logits[1].flip(-1)
        logits = get_numpy_from_tensor(logits).transpose((1, 2, 0))
        return logits

    with torch.no_grad():
        length = len(dataset)
        for step, (ori_image, image_id, gt_mask) in enumerate(dataset):
            ori_w, ori_h = ori_image.size

            cams_list = []

            for scale in scales:
                image = copy.deepcopy(ori_image)
                image = image.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.CUBIC)
                
                image = normalize_fn(image)
                image = image.transpose((2, 0, 1))

                image = torch.from_numpy(image)
                flipped_image = image.flip(-1)
                
                images = torch.stack([image, flipped_image])

                cams = inference(images, (ori_h, ori_w))
                cams_list.append(cams)
            
            preds = np.sum(cams_list, axis=0)
            preds = F.softmax(torch.from_numpy(preds), dim=-1).numpy()
            
            if args.iteration > 0:
                # h, w, c -> c, h, w
                preds = crf_inference(np.asarray(ori_image), preds.transpose((2, 0, 1)), t=args.iteration)
                pred_mask = np.argmax(preds, axis=0)
            else:
                pred_mask = np.argmax(preds, axis=-1)
            #     # 定义标签颜色映射
            #     # # 定义标签颜色映射
            #     # label_colors = {
            #     #     0: (0, 0, 0),  # 背景
            #     #     1: (0, 63, 255),  # 类别1
            #     #     2: (0, 127, 191),  # 类别2
            #     #     3: (0, 0, 127),  # 类别3
            #     #     4: (0, 0, 63),  # 类别3
            #     #     5: (0, 127, 63),  # 类别3
            #     #     6: (0, 63, 191),  # 类别3
            #     #     7: (0, 63, 0),  # 类别3
            #     #     8: (0, 191, 127),  # 类别3
            #     #     9: (0, 127, 255),  # 类别3
            #     #     10: (0, 63, 127),  # 类别3
            #     #     11: (0, 127, 127),  # 类别3
            #     #     12: (0, 63, 63),  # 类别3
            #     #     13: (0, 100, 155),  # 类别3
            #     #     14: (0, 0, 255),  # 类别3
            #     #     15: (0, 0, 191)  # 类别3
            #     # }
            #
            #     label_colors = {
            #         0: (255, 255, 255),  # 背景
            #         1: (0, 0, 255),  # 类别1
            #         2: (0, 255, 255),  # 类别2
            #         3: (0, 255, 0),  # 类别3
            #         4: (255, 204, 0),  # 类别3
            #         5: (255, 0, 0)  # 类别3
            #     }
            #     # label_colors = {
            #     #     0: (128, 128, 128),  # 背景
            #     #     1: (255, 0, 0),  # 类别1
            #     #     2: (192, 192, 0),  # 类别2
            #     #     3: (255, 255, 0),  # 类别3
            #     #     4: (0, 255, 0),  # 类别3
            #     #     5: (0, 0, 255)  # 类别3
            #     # }
            #     # 创建一个彩色图像，与标签掩码图像大小相同
            #     colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
            #
            #     # 遍历标签掩码图像的每个像素，并根据标签颜色映射进行上色
            #     for label in label_colors:
            #         colored_mask[pred_mask == label] = label_colors[label]
            #
            # imageio.imwrite(pred_vis_dir + image_id + '.png', colored_mask.astype(np.uint8))
            imageio.imwrite(pred_dir + image_id + '.png', pred_mask.astype(np.uint8))

            sys.stdout.write('\r# Make CAM [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
            sys.stdout.flush()
        print()
    
    if args.domain == 'val':
        print("python3 evaluate.py --experiment_name {} --domain {} --mode png".format(args.tag, args.domain))