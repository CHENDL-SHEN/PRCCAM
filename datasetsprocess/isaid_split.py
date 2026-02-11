import glob
import os
import numpy as np
import cv2
from PIL import Image
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu
from torchvision.transforms import (Pad, ColorJitter, Resize, FiveCrop, RandomCrop,
                                    RandomHorizontalFlip, RandomRotation, RandomVerticalFlip)
import random

SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# 该颜色列表是BGR颜色数值，并不是RGB
# ImSurf = np.array([255, 255, 255])  # label 0
# Building = np.array([255, 0, 0]) # label 1
# LowVeg = np.array([255, 255, 0]) # label 2
# Tree = np.array([0, 255, 0]) # label 3
# Car = np.array([0, 255, 255]) # label 4
# Clutter = np.array([0, 0, 255]) # label 5
# Boundary = np.array([0, 0, 0]) # label 6
# num_classes = 6

Background =  np.array([0, 0, 0])
GTF = np.array([255, 63, 0])  # label 1
SBF = np.array([191, 127, 0]) # label 2
SV = np.array([127, 0, 0]) # label 3
Ship = np.array([63, 0, 0]) # label 4
Bridge = np.array([63, 127, 0]) # label 5
BC = np.array([191, 63, 0]) # label 6
BD = np.array([0, 63, 0]) # label 7
RA = np.array([127, 191, 0]) # label 8
Plane = np.array([255, 127, 0]) # label 9
TC = np.array([127, 63, 0]) # label 10
LV = np.array([127, 127, 0]) # label 11
ST = np.array([63, 63, 0]) # label 12
Harbor = np.array([155, 100, 0]) # label 13
SP = np.array([255, 0, 0]) # label 14
HC = np.array([191, 0, 0]) # label 15

num_classes = 16

# split huge RS image to small patches
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", default="/media/ders/sdc1/FJM/datasets/Generality/I_fen/origion5/train/images/")
    parser.add_argument("--mask-dir", default="/media/ders/sdc1/FJM/datasets/Generality/I_fen/origion5/train/labels/")
    parser.add_argument("--output-img-dir", default="/media/ders/sdc1/FJM/datasets/Generality/I512s512/crop5/train/images/")
    parser.add_argument("--output-mask-dir", default="/media/ders/sdc1/FJM/datasets/Generality/I512s512/crop5/train/labels/")
    parser.add_argument("--vis_output_mask_dir", default="/media/ders/sdc1/FJM/datasets/Generality/I512s512/crop5/train/labels_vis/")
    # parser.add_argument("--eroded", default=True)
    parser.add_argument("--eroded", action='store_true')
    parser.add_argument("--gt", action='store_true')
    # parser.add_argument("--gt", default=True)
    parser.add_argument("--mode", type=str, default='val')
    parser.add_argument("--val-scale", type=float, default=1.0)
    parser.add_argument("--split-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    return parser.parse_args()


def get_img_mask_padded(image, mask, patch_size, mode):
    img_pad = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # RGB转化为BGR图片
    mask_pad = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)  # 红色通道和蓝色通道调换位置 mask像素为 0 0 6
    # img, mask = np.array(image), np.array(mask)  # H*W*3  RGB图像变为数组 三列形式 每一行是三个通道的像素值；mask标签就是三通道的像素值
    # oh, ow = img.shape[0], img.shape[1]  # 2557 1887
    # rh, rw = oh % patch_size, ow % patch_size  # 取完整数个裁剪图片尺寸后剩下的尺寸（宽度和高度）
    # width_pad = 0 if rw == 0 else patch_size - rw  # 512-351=161要填充的像素宽度数
    # height_pad = 0 if rh == 0 else patch_size - rh  # 3
    #
    # h, w = oh + height_pad, ow + width_pad  # 2048*2560
    # pad_img = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right', border_mode=cv2.BORDER_CONSTANT, value=0)(image=img)  # 原始图片放在右下角，在右下角网往左上角用0填充原始图像
    # pad_mask = albu.PadIfNeeded(min_height=h, min_width=w, position='bottom_right', border_mode=cv2.BORDER_CONSTANT, value=255)(image=mask)  # 在右下角用标签6填充原始图像,三维表示形式改为255是为了和背景像素相同，填充的部分作为背景
    # img_pad, mask_pad = pad_img['image'], pad_mask['image']  # mask像素为6 0 0
    # img_pad = cv2.cvtColor(np.array(img_pad), cv2.COLOR_RGB2BGR)  # RGB转化为BGR图片
    # mask_pad = cv2.cvtColor(np.array(mask_pad), cv2.COLOR_RGB2BGR)  # 红色通道和蓝色通道调换位置 mask像素为 0 0 6
    return img_pad, mask_pad


# def pv2rgb(mask):
#     h, w = mask.shape[0], mask.shape[1]
#     mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
#     mask_convert = mask[np.newaxis, :, :]
#     mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
#     mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
#     mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
#     mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
#     mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
#     mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
#     return mask_rgb
def pv2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [127, 0, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 0, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 63, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [191, 127, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [63, 0, 0]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [63, 127, 0]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [191, 63, 0]
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [0, 63, 0]
    mask_rgb[np.all(mask_convert == 8, axis=0)] = [127, 191, 0]
    mask_rgb[np.all(mask_convert == 9, axis=0)] = [255, 127, 0]
    mask_rgb[np.all(mask_convert == 10, axis=0)] = [127, 63, 0]
    mask_rgb[np.all(mask_convert == 11, axis=0)] = [127, 127, 0]
    mask_rgb[np.all(mask_convert == 12, axis=0)] = [63, 63, 0]
    mask_rgb[np.all(mask_convert == 13, axis=0)] = [155, 100, 0]
    mask_rgb[np.all(mask_convert == 14, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 15, axis=0)] = [191, 0, 0]
    return mask_rgb

def car_color_replace(mask):
    mask = cv2.cvtColor(np.array(mask.copy()), cv2.COLOR_RGB2BGR)
    mask[np.all(mask == [0, 255, 255], axis=-1)] = [0, 204, 255]

    return mask


# def rgb_to_2D_label(_label):
#     _label = _label.transpose(2, 0, 1)  # Label现在是BGR图片，为H*W*C，经过 2 0 1 后变为C*H*W图片 就是分为3个通道 每一个通道每一个通道那么显示
#     label_seg = np.zeros(_label.shape[1:], dtype=np.uint8)  # 上边转换为CHW为了在这里将C置为1，[1:]指的是CHW中的H*W维度
#     label_seg[np.all(_label.transpose([1, 2, 0]) == ImSurf, axis=-1)] = 0  # 当前label是BGR格式，类别对应的颜色也是BGR格式颜色
#     label_seg[np.all(_label.transpose([1, 2, 0]) == Building, axis=-1)] = 1
#     label_seg[np.all(_label.transpose([1, 2, 0]) == LowVeg, axis=-1)] = 2
#     label_seg[np.all(_label.transpose([1, 2, 0]) == Tree, axis=-1)] = 3
#     label_seg[np.all(_label.transpose([1, 2, 0]) == Car, axis=-1)] = 4
#     label_seg[np.all(_label.transpose([1, 2, 0]) == Clutter, axis=-1)] = 5
#     label_seg[np.all(_label.transpose([1, 2, 0]) == Boundary, axis=-1)] = 6
#     return label_seg  # 填充部分的0 0 6部分在创建数组时候归为0了
def rgb_to_2D_label(_label):
    _label = _label.transpose(2, 0, 1)  # Label现在是BGR图片，为H*W*C，经过 2 0 1 后变为C*H*W图片 就是分为3个通道 每一个通道每一个通道那么显示
    label_seg = np.zeros(_label.shape[1:], dtype=np.uint8)  # 上边转换为CHW为了在这里将C置为1，[1:]指的是CHW中的H*W维度
    label_seg[np.all(_label.transpose([1, 2, 0]) == Background, axis=-1)] = 0  # 当前label是BGR格式，类别对应的颜色也是BGR格式颜色
    label_seg[np.all(_label.transpose([1, 2, 0]) == GTF, axis=-1)] = 1
    label_seg[np.all(_label.transpose([1, 2, 0]) == SBF, axis=-1)] = 2
    label_seg[np.all(_label.transpose([1, 2, 0]) == SV, axis=-1)] = 3
    label_seg[np.all(_label.transpose([1, 2, 0]) == Ship, axis=-1)] = 4
    label_seg[np.all(_label.transpose([1, 2, 0]) == Bridge, axis=-1)] = 5
    label_seg[np.all(_label.transpose([1, 2, 0]) == BC, axis=-1)] = 6
    label_seg[np.all(_label.transpose([1, 2, 0]) == BD, axis=-1)] = 7
    label_seg[np.all(_label.transpose([1, 2, 0]) == RA, axis=-1)] = 8
    label_seg[np.all(_label.transpose([1, 2, 0]) == Plane, axis=-1)] = 9
    label_seg[np.all(_label.transpose([1, 2, 0]) == TC, axis=-1)] = 10
    label_seg[np.all(_label.transpose([1, 2, 0]) == LV, axis=-1)] = 11
    label_seg[np.all(_label.transpose([1, 2, 0]) == ST, axis=-1)] = 12
    label_seg[np.all(_label.transpose([1, 2, 0]) == Harbor, axis=-1)] = 13
    label_seg[np.all(_label.transpose([1, 2, 0]) == SP, axis=-1)] = 14
    label_seg[np.all(_label.transpose([1, 2, 0]) == HC, axis=-1)] = 15
    return label_seg  # 填充部分的0 0 6部分在创建数组时候归为0了

def image_augment(image, mask, patch_size, mode='train', val_scale=1.0):
    image_list = []  # RGB
    mask_list = []  # RGB
    image_width, image_height = image.size[1], image.size[0]
    mask_width, mask_height = mask.size[1], mask.size[0]

    assert image_height == mask_height and image_width == mask_width
    if mode == 'train':
        # # resize_0 = Resize(size=(int(image_width * 0.25), int(image_height * 0.25)))
        # # resize_1 = Resize(size=(int(image_width * 0.5), int(image_height * 0.5)))
        # # resize_2 = Resize(size=(int(image_width * 0.75), int(image_height * 0.75)))
        # # resize_3 = Resize(size=(int(image_width * 1.25), int(image_height * 1.25)))
        # # resize_4 = Resize(size=(int(image_width * 1.5), int(image_height * 1.5)))
        # # resize_5 = Resize(size=(int(image_width * 1.75), int(image_height * 1.75)))
        # # resize_6 = Resize(size=(int(image_width * 2.0), int(image_height * 2.0)))
        # # image_resize_0, mask_resize_0 = resize_0(image.copy()), resize_0(mask.copy())
        # # image_resize_1, mask_resize_1 = resize_1(image.copy()), resize_1(mask.copy())
        # # image_resize_2, mask_resize_2 = resize_2(image.copy()), resize_2(mask.copy())
        # # image_resize_3, mask_resize_3 = resize_3(image.copy()), resize_3(mask.copy())
        # # image_resize_4, mask_resize_4 = resize_4(image.copy()), resize_4(mask.copy())
        # # image_resize_5, mask_resize_5 = resize_5(image.copy()), resize_5(mask.copy())
        # # image_resize_6, mask_resize_6 = resize_6(image.copy()), resize_6(mask.copy())
        # h_vlip = RandomHorizontalFlip(p=1.0)
        # v_vlip = RandomVerticalFlip(p=1.0)
        # # crop_1 = RandomCrop(size=(int(image_width*0.75), int(image_height*0.75)))
        # # crop_2 = RandomCrop(size=(int(image_width * 0.5), int(image_height * 0.5)))
        # # color = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        # image_h_vlip, mask_h_vlip = h_vlip(image.copy()), h_vlip(mask.copy())
        # image_v_vlip, mask_v_vlip = v_vlip(image.copy()), v_vlip(mask.copy())
        # # image_crop_1, mask_crop_1 = crop_1(image.copy()), crop_1(mask.copy())
        # # image_crop_2, mask_crop_2 = crop_2(image.copy()), crop_2(mask.copy())
        # # image_color = color(image.copy())
        #
        # image_list_train = [image, image_h_vlip, image_v_vlip]
        # mask_list_train = [mask, mask_h_vlip, mask_v_vlip]
        image_list_train = [image]
        mask_list_train = [mask]
        for i in range(len(image_list_train)):
            image_tmp, mask_tmp = get_img_mask_padded(image_list_train[i], mask_list_train[i], patch_size, mode)
            mask_tmp = rgb_to_2D_label(mask_tmp.copy())
            image_list.append(image_tmp)
            mask_list.append(mask_tmp)
    else:
        rescale = Resize(size=(int(image_width * val_scale), int(image_height * val_scale)))
        image, mask = rescale(image.copy()), rescale(mask.copy())
        image, mask = get_img_mask_padded(image.copy(), mask.copy(), patch_size, mode)
        mask = rgb_to_2D_label(mask.copy())
        image_list.append(image)
        mask_list.append(mask)
    return image_list, mask_list


def randomsizedcrop(image, mask):
    # assert image.shape[:2] == mask.shape
    h, w = image.shape[0], image.shape[1]
    crop = albu.RandomSizedCrop(min_max_height=(int(3*h//8), int(h//2)), width=h, height=w)(image=image.copy(), mask=mask.copy())
    img_crop, mask_crop = crop['image'], crop['mask']  # 裁剪的时候宽高相同都是在(int(3*h//8), int(h//2))之间的随机尺寸，先裁剪然后在缩放至512*512
    return img_crop, mask_crop


def car_aug(image, mask):
    assert image.shape[:2] == mask.shape
    v_flip = albu.VerticalFlip(p=1.0)(image=image.copy(), mask=mask.copy())
    h_flip = albu.HorizontalFlip(p=1.0)(image=image.copy(), mask=mask.copy())
    rotate_90 = albu.RandomRotate90(p=1.0)(image=image.copy(), mask=mask.copy())
    # blur = albu.GaussianBlur(p=1.0)(image=image.copy())
    image_vflip, mask_vflip = v_flip['image'], v_flip['mask']
    image_hflip, mask_hflip = h_flip['image'], h_flip['mask']
    image_rotate, mask_rotate = rotate_90['image'], rotate_90['mask']
    # blur_image = blur['image']
    image_list = [image, image_vflip, image_hflip, image_rotate]
    mask_list = [mask, mask_vflip, mask_hflip, mask_rotate]

    return image_list, mask_list


def vaihingen_format(inp):
    (img_path, mask_path, imgs_output_dir, masks_output_dir, vis_masks_output_dir, eroded, gt, mode, val_scale, split_size, stride) = inp
    img_filename = os.path.splitext(os.path.basename(img_path))[0]  # [0]指的是提取文件名，[1]指的是后缀“.tif”
    mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
    # if eroded:
    #     mask_path = mask_path[:-4] + '_noBoundary.tif'
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('RGB')
    # if gt:
    #     mask_ = car_color_replace(mask)
    #     out_origin_mask_path = os.path.join(masks_output_dir + '/origin/', "{}.tif".format(mask_filename))
    #     cv2.imwrite(out_origin_mask_path, mask_)
    image_pad = img.copy()
    img_list = cv2.cvtColor(np.array(image_pad), cv2.COLOR_RGB2BGR)  # RGB转化为BGR图片
    mask_pad = cv2.cvtColor(np.array(mask.copy()), cv2.COLOR_RGB2BGR)
    mask_list = rgb_to_2D_label(mask_pad.copy())
    # print(img_path)
    # print(img.size, mask.size)
    # img and mask shape: WxHxC
    # image_list, mask_list = image_augment(image=img.copy(), mask=mask.copy(), patch_size=split_size,
    #                                       mode=mode, val_scale=val_scale)
    assert img_filename+ '_instance_color_RGB' == mask_filename
    image_number = random.choice([6, 7, 8])
    # for m in range(len(image_list)):
    k = 0
    img = img_list  # 取图片
    mask = mask_list  # 取标签
    assert img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1]
    # if gt:
    #     mask = pv2rgb(mask)
    used_positions = set()

    for iter in range(100) :
        if img.shape[0] <= split_size or img.shape[1] <= split_size:
            continue
        y = random.randrange(0, img.shape[0] - split_size)
        x = random.randrange(0, img.shape[1] - split_size)
        if (x, y) in used_positions:
            continue
        img_tile = img[y:y + split_size, x:x + split_size]  # 裁剪下来的图片变量
        mask_tile = mask[y:y + split_size, x:x + split_size]  # 裁剪下来的标签变量
        bins = np.array(range(num_classes))  # 创建一个包含num_classes + 1个元素的数组bins，其中元素的值从0到num_classes
        class_pixel_counts, _ = np.histogram(mask_tile.copy(), bins=bins)  # class_pixel_counts指的是[0,1)区间也就是0的个数
        cf = class_pixel_counts / (mask_tile.copy().shape[0] * mask_tile.copy().shape[1])  # 每一个类别对应的概率
        if cf[0] < 0.98:  # test 0.9;train0.98
            mask_tile_vis = mask_tile
            if img_tile.shape[0] == split_size and img_tile.shape[1] == split_size \
                    and mask_tile.shape[0] == split_size and mask_tile.shape[1] == split_size:
                out_img_path = os.path.join(imgs_output_dir, "{}_{}.tif".format(img_filename, k))
                cv2.imwrite(out_img_path,
                            img_tile)  # 存的是BGR格式图片，在训练网络加载图片时候先转换为RGB格式img = Image.open(img_name).convert('RGB')

                out_mask_path = os.path.join(masks_output_dir, "{}_{}.png".format(img_filename, k))
                cv2.imwrite(out_mask_path,
                            mask_tile)  # 存的是BGR格式图片，在训练网络加载图片时候先转换为灰度格式mask = Image.open(mask_name).convert('L')
                # 可视化
                mask_vis = pv2rgb(mask_tile_vis)
                vis_out_mask_path = os.path.join(vis_masks_output_dir, "{}_{}.png".format(img_filename, k))
                cv2.imwrite(vis_out_mask_path, mask_vis)
                k += 1
                if k >= image_number:
                    break
        else:
            continue


if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    imgs_dir = args.img_dir
    masks_dir = args.mask_dir
    imgs_output_dir = args.output_img_dir
    masks_output_dir = args.output_mask_dir
    vis_masks_output_dir = args.vis_output_mask_dir
    gt = args.gt
    # gt = True
    # eroded = True
    eroded = args.eroded
    mode = args.mode
    val_scale = args.val_scale
    split_size = args.split_size
    stride = args.stride
    img_paths = glob.glob(os.path.join(imgs_dir, "*.png"))  # 查找指定目录下所有以.tif结尾的文件。使用glob模块来获取指定文件夹imgs_dir下所有以.tif结尾的文件的路径，并将这些路径存储在img_paths变量中。
    mask_paths_raw = glob.glob(os.path.join(masks_dir, "*.png"))
    if eroded:
        mask_paths = [(p[:-15] + '.tif') for p in mask_paths_raw]
    else:
        mask_paths = mask_paths_raw
    img_paths.sort()  # 这行代码将img_paths列表中的元素按照默认的字典顺序进行排序，以便后续保持图片和标签保持一致
    mask_paths.sort()

    if not os.path.exists(imgs_output_dir):
        os.makedirs(imgs_output_dir)
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)
    if not os.path.exists(vis_masks_output_dir):
        os.makedirs(vis_masks_output_dir)
        if gt:
            os.makedirs(masks_output_dir+'/origin')

    inp = [(img_path, mask_path, imgs_output_dir, masks_output_dir, vis_masks_output_dir, eroded, gt, mode, val_scale, split_size, stride)
           for img_path, mask_path in zip(img_paths, mask_paths)]

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(vaihingen_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print('images spliting spends: {} s'.format(split_time))


