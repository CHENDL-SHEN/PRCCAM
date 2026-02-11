import glob
import os
import numpy as np
import cv2
from PIL import ImageEnhance, Image
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu
from torchvision.transforms import (Pad, ColorJitter, Resize, FiveCrop, RandomCrop,
                                    RandomHorizontalFlip, RandomRotation, RandomVerticalFlip)
import random
import shutil
import imageio
import tqdm
import matplotlib.pyplot as plt
SEED = 42
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def calculate_iou(pred_mask, true_mask, num_classes):
    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)

    for class_idx in range(num_classes):
        pred_class = pred_mask == class_idx
        true_class = true_mask == class_idx

        intersection[class_idx] = np.sum(np.logical_and(pred_class, true_class))
        union[class_idx] = np.sum(np.logical_or(pred_class, true_class))

    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)
    return iou

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# num_classes = 6

# split huge RS image to small patches
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default="/media/ders/sdc1/FJM/datasets/Paper3mul/Is512s512/train/images/")  # 原始图像
    parser.add_argument("--mask_dir", default="/media/ders/sdc1/FJM/datasets/Paper3mul/Is512s512/train/labels/")  # 真是标签
    parser.add_argument("--mask_npy_dir",
                        default="/media/ders/sdc1/FJM/datasets/Paper3mul/Is512s512/train/classification/")
    parser.add_argument("--pseudo_label_dir", default="/media/ders/sdc1/FJM/115P4mul_ipc_AP_aff/experimentsI512s512_SFC/pseudo_labels_0.6_best/label/")  # 伪标签
    parser.add_argument("--sam_mask_dir", default="/media/ders/sdc1/FJM/P3_S2C_SAM/I512s512_train/")

    # parser.add_argument("--output_pseudo_dir", default="/media/ders/sdc1/FJM/115P4mul_ipc_AP_SAM/Refined_pseudo_labels_SFC0.85/V128s128/pseudo/")  # puzzle伪标签
    parser.add_argument("--output_refined_pseudo_dir", default="/media/ders/sdc1/FJM/115P4mul_ipc_AP_SAM/experimentsI512s512_SFC/label_th0.9/")  # ours伪标签
    # parser.add_argument("--output_refined_pseudo_vis_dir", default="/media/ders/sdc1/FJM/115P4mul_ipc_AP_SAM/Refined_pseudo_labels_SFC0.85/V128s128/refined_pseudo_vis/")  # ours伪标签

    # parser.add_argument("--output_label_dir", default="/media/ders/sdc1/FJM/115P4mul_ipc_AP_SAM/Refined_pseudo_labels_SFC0.85/V128s128/label/")  # ours伪标签
    # parser.add_argument("--output_image_dir", default="/media/ders/sdc1/FJM/115P4mul_ipc_AP_SAM/Refined_pseudo_labels_SFC0.85/V128s128/images/")
    parser.add_argument('--th', default=0.9, type=float)
    return parser.parse_args()


# def pv2rgb(mask):  # WHDLD数据集颜色
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
def pv2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 255]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 255, 255]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [255, 204, 0]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 255, axis=0)] = [0, 0, 0]
    return mask_rgb

def compare_labels(label1, label2):
    # 获取两个标签中值为1的位置
    positions1 = set(zip(*np.where(label1 == 1)))
    positions2 = set(zip(*np.where(label2 == 1)))

    # 统计信息
    count1 = len(positions1)
    count2 = len(positions2)
    common_positions = positions1 & positions2  # 交集
    unique_to_label1 = positions1 - positions2  # 只在label1中的位置
    unique_to_label2 = positions2 - positions1  # 只在label2中的位置

    print(f"标签1中为1的像素数量: {count1}")
    print(f"标签2中为1的像素数量: {count2}")
    print(f"相同位置都为1的像素数量: {len(common_positions)}")
    print(f"只在标签1中为1的像素数量: {len(unique_to_label1)}")
    print(f"只在标签2中为1的像素数量: {len(unique_to_label2)}")

    # 判断是否完全相同
    if positions1 == positions2:
        print("结果: 相同 - 两个标签中为1的位置完全一致")
    else:
        print("结果: 不同 - 两个标签中为1的位置存在差异")

    return common_positions, unique_to_label1, unique_to_label2

if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    imgs_dir = args.img_dir
    masks_dir = args.mask_dir
    pseudo_labels_dir = args.pseudo_label_dir
    sam_masks_dir = args.sam_mask_dir

    masks_npy_dir = args.mask_npy_dir
    label_npy = masks_npy_dir + 'train_.npy'  # 存标签图片的文件夹
    labels_npy_dict = np.load(label_npy, allow_pickle=True).item()

    # pseudo_output_dir = args.output_pseudo_dir
    refined_pseudo_output_dir = args.output_refined_pseudo_dir
    # refined_pseudo_vis_output_dir = args.output_refined_pseudo_vis_dir
    # label_output_dir = args.output_label_dir
    # image_output_dir = args.output_image_dir

    # if not os.path.exists(pseudo_output_dir):
    #     os.makedirs(pseudo_output_dir)
    if not os.path.exists(refined_pseudo_output_dir):
        os.makedirs(refined_pseudo_output_dir)
    # if not os.path.exists(refined_pseudo_vis_output_dir):
    #     os.makedirs(refined_pseudo_vis_output_dir)
    # if not os.path.exists(label_output_dir):
    #     os.makedirs(label_output_dir)
    # if not os.path.exists(image_output_dir):
    #     os.makedirs(image_output_dir)
    #读取文件
    img_paths = glob.glob(os.path.join(imgs_dir, "*.tif"))  # 查找指定目录下所有以.tif结尾的文件。使用glob模块来获取指定文件夹imgs_dir下所有以.tif结尾的文件的路径，并将这些路径存储在img_paths变量中。
    mask_paths = glob.glob(os.path.join(masks_dir, "*.png"))
    pseudo_masks_paths = glob.glob(os.path.join(pseudo_labels_dir, "*.png"))
    sam_masks_paths = glob.glob(os.path.join(sam_masks_dir, "*.npy"))


    img_paths.sort()  # 这行代码将img_paths列表中的元素按照默认的字典顺序进行排序，以便后续保持图片和标签保持一致
    mask_paths.sort()
    pseudo_masks_paths.sort()
    sam_masks_paths.sort()
    # filter_masks_paths.sort()
    # mask = Image.open(mask_paths)  # 真实标签
    for img_path in tqdm.tqdm(img_paths):
        img_filename = os.path.splitext(os.path.basename(img_path))[0]
        img_save_path = os.path.join(imgs_dir, "{}.tif".format(img_filename))
        img = Image.open(img_save_path).convert('RGB')
        img_array = np.array(img)

        keys = labels_npy_dict.get(img_filename, 'Key not found')
        indices = np.flatnonzero(keys)
        # non_zero_count = sum(1 for x in keys if x != 0)
        # num_classes = non_zero_count  # 该图像的类别标签

        # ori_label = os.path.join(masks_dir, "{}.png".format(img_filename))
        # label = Image.open(ori_label)
        # label_array = np.array(label)

        pseudo_labels = os.path.join(pseudo_labels_dir, "{}.png".format(img_filename))
        pseudo = Image.open(pseudo_labels)
        pseudo_array = np.array(pseudo)
        pseudo_refined_array = pseudo_array.copy()  # 复制原数组


        sam_npy = sam_masks_dir + "{}.npy".format(img_filename)  # 存标签图片的文件夹
        sam_npy_dict = np.load(sam_npy, allow_pickle=True)
        numeric_sams = []
        for i, mask in enumerate(sam_npy_dict):
            numeric_masks = sam_npy_dict[i]
            bool_mask = numeric_masks['segmentation']
            numeric_masks['segmentation'] = np.where(bool_mask, 1, 0)
            numeric_sams.append(numeric_masks)
        # numeric_masks = [(mask * 1).astype(np.uint8) for mask in sam_npy_dict['masks']]

        # unique_labels = np.unique(pseudo_array)
       # print(f"伪标签中包含的标签值: {unique_labels}")

        for label_value in indices:
            # 创建当前标签的掩码
            pseudo_i255_mask = np.where(pseudo_array == label_value, 1, 255).astype(np.uint8)  # 伪标签对应的i类别，对于这个类别重新做一个目标区域为1，其余为0的
            # pseudo_refined_array = pseudo_i255_mask.copy()
            # ################保存每个类别的伪标签#############
            # pseudo_i255 = pseudo_i255_mask.copy()
            # pseudo_i255_save = pseudo_output_dir + img_filename + "_{}.png".format(label_value)
            # imageio.imwrite(pseudo_i255_save, pseudo_i255.astype(np.uint8))

            pseudo_count = np.sum(pseudo_i255_mask == 1)
            #  print(f"pseudo中1的个数: {pseudo_count}")
            for i, mask in enumerate(numeric_sams):
                sam_labels = numeric_sams[i]
                sam_result = sam_labels['segmentation']

                sam_result_count = np.sum(sam_result == 1)
                # common, unique1, unique2 = compare_labels(pseudo_i255_mask, sam_result)
                intersection_mask = np.logical_and(pseudo_i255_mask == 1, sam_result == 1).astype(np.uint8)
                # intersection_mask = pseudo_i255_mask & sam_result
               # print(f"sam中1的个数: {sam_result_count}")
                intersection = np.sum(intersection_mask)
               # print(f"两个数组中同时为1的像素个数: {intersection}")
                if intersection != 0:
                    iou_p = intersection / pseudo_count
                    iou_sam = intersection / sam_result_count
                    if iou_p >= args.th or iou_sam >= args.th:
                        if iou_p >= iou_sam:
                            mask_A_ones = (pseudo_i255_mask == 1)
                            pseudo_refined_array[mask_A_ones] = label_value
                        else:
                            mask_B_ones = (sam_result == 1)
                            pseudo_refined_array[mask_B_ones] = label_value
                    else:
                        continue
                else:
                    continue
                # ########精炼后单类别的可视化效果###########
                # mask = (pseudo_refined_array == label_value)
                # processed_mask[mask] = label_value
                # pseudo_refined_single_vis = pseudo_refined_array.copy()
                # pseudo_refined_save = refined_pseudo_vis_output_dir + img_filename + "_{}.png".format(label_value)
                # imageio.imwrite(pseudo_refined_save, pseudo_refined_single_vis.astype(np.uint8))



        pseudo_refined_label = pseudo_refined_array
        pseudo_array_save = refined_pseudo_output_dir + img_filename + '.png'
        imageio.imwrite(pseudo_array_save, pseudo_refined_label.astype(np.uint8))

        # pseudo_refined_vis = pv2rgb(pseudo_refined_array.copy())
        # pseudo_refined_save = refined_pseudo_vis_output_dir + img_filename + '.png'
        # imageio.imwrite(pseudo_refined_save, pseudo_refined_vis.astype(np.uint8))

        # pseudo_vis = pv2rgb(pseudo_array)
        # pseudo_save = pseudo_output_dir + img_filename + '.png'
        # imageio.imwrite(pseudo_save, pseudo_vis.astype(np.uint8))

        # label_vis = pv2rgb(label_array)
        # label_save = label_output_dir + img_filename + '.png'
        # imageio.imwrite(label_save, label_vis.astype(np.uint8))
        #
        # image_save = image_output_dir + img_filename + '.tif'
        # imageio.imwrite(image_save, img_array.astype(np.uint8))
