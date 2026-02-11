import os
import numpy as np
import argparse
import json
from PIL import Image
from os.path import join


# 设标签宽W，长H
def fast_hist(a, b, n):  # a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)；n是类别数目，实数（在这里为19）
    k = (a >= 0) & (a < n) & (a != 255)  # k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景） k=0或1
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)

def per_class_iu(hist):  # 分别为每个类别（在这里是19类）计算mIoU，hist的形状(n, n)
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  # 矩阵的对角线上的值组成的一维数组/矩阵的所有元素之和，返回值形状(n,)
def get_tp_fp_tn_fn(hist):
    tp = np.diag(hist)# 标签为正，预测也为正，就是斜对角上的元素
    fp = hist.sum(0) - np.diag(hist)# 负类预测为正类，axis=0每列元素的和。
    fn = hist.sum(1) - np.diag(hist)# 正类预测为负类，axis=1每行元素求和。
    tn = hist.sum(1) + hist.sum(0) - np.diag(hist)# 负类预测为负类，sum()所有元素求和。
    return tp, fp, tn, fn
# hist.sum(0)=按列相加  hist.sum(1)按行相加
def F1ger(hist):
    tp, fp, tn, fn = get_tp_fp_tn_fn(hist)
    Precision = tp / (tp + fp)
    Recall = tp / (tp + fn)
    F1 = (2.0 * Precision * Recall) / (Precision + Recall)
    return F1

def compute_mIoU(gt_dir, pred_dir):  # 计算mIoU的函数

    num_classes = 16
    print('Num classes', num_classes)
    name_classes = ["BG", "GTF", "SBF", "SV", "SH", "BR", "BC", "BD", "RA", "PL", "TC", "LV", "ST", "HA", "SP", "HC"]
    # num_classes = 5
    # print('Num classes', num_classes)
    # # name_classes = ["ImSurf","Building","LowVeg","Tree","Car","Clutter"]
    # name_classes = ["ImSurf","Building","LowVeg","Tree","Car"]
    hist = np.zeros((num_classes, num_classes))

    image_data_list = os.listdir(gt_dir)  # 带后缀的所有文件名
    gt_imgs = [it[:-4] for it in image_data_list]  # 不带后缀的所有文件名
    gt_imgs.sort()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]  # 获得验证集标签路径列表，方便直接读取

    pred_data_list = os.listdir(pred_dir)  # 带后缀的所有文件名
    pred_imgs = [it[:-4] for it in pred_data_list]  # 不带后缀的所有文件名
    pred_imgs.sort()
    pred_imgs = [join(pred_dir, x) for x in pred_imgs]

    for ind in range(len(gt_imgs)):  # 读取每一个（图片-标签）对
        pred = np.array(Image.open(pred_imgs[ind]+'.png'))  # 读取一张图像分割结果，转化成numpy数组
        label = np.array(Image.open(gt_imgs[ind]+'.png'))  # 读取一张对应的标签，转化成numpy数组
        mask = (pred != 16)  # 创建一个掩码，忽略类别为6的部分
        pred = pred[mask]
        label = label[mask]
        if len(label.flatten()) != len(pred.flatten()):  # 如果图像分割结果与标签的大小不一样，这张图片就不计算
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()),
                                                                                  len(pred.flatten()), gt_imgs[ind],
                                                                                  pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)  # 对一张图片计算19×19的hist矩阵，并累加
        if ind > 0 and ind % 10 == 0:  # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.mean(per_class_iu(hist))))
            print(per_class_iu(hist))

    mIoUs = per_class_iu(hist)  # 计算所有验证集图片的逐类别mIoU值
    F1 = F1ger(hist)  # 计算所有验证集图片的逐类别mIoU值
    for ind_class in range(num_classes):  # 逐类别输出一下mIoU值72.51
        print('===>' + name_classes[ind_class] + 'IOU' + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
        # print('===>' + name_classes[ind_class] + 'F1' + ':\t' + str(round(F1[ind_class] * 100, 2)))

    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))  # 在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    print('===> AverageF1: ' + str(round(np.nanmean(F1) * 100, 2)))
    return mIoUs

compute_mIoU('/media/ders/sdd1/FJM/datasets/Paper3mul/Is512s512/test/labels/',
             '/media/ders/sdc1/FJM/115P4mul_ipc_AP_aff/experimentsI512s512_SFC/segmentation/predictions/pred_label/',
             )
