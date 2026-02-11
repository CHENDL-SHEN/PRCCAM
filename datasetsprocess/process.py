import glob
import numpy as np
import cv2
import os
from tqdm import tqdm


def pixel_label_to_tag(data_dir,save_path, num_classes,  delete_ignore=True):
    if not os.path.exists(data_dir): raise RuntimeError('data_dir do not exists！')
    images_dir = glob.glob(os.path.join(data_dir,'*.png'))
    label = {}      # image-level-label dict
    for n, image_dir in enumerate(tqdm(images_dir)):
        image_name = os.path.splitext(os.path.basename(image_dir))[0]
        image = cv2.imread(image_dir)
        image_classes_ori = np.unique(image)  # np.unique() 函数 去除其中重复的元素 ，并按元素 由小到大 返回一个新的无元素重复的元组或者列表。
        # image_classes = [i for i in image_classes_ori if (not i in ignore_cls) and i!=255 ]
        # image_classes = [i for i in image_classes_ori if (not i in ignore_cls)]
        image_classes = [i for i in image_classes_ori]
        labelarr = np.zeros(num_classes)
        flag = False
        for i in image_classes:
            # if i==255: continue
            '''dataset filtering'''
            # if i==5:
            #     if np.sum(image == i) < 2048:
            #         flag = True
            #         break
            # else:
            #     if np.sum(image == i) < 8192:
            #         flag = True
            #         break
            ''''''
            labelarr[i]=1
        if not flag:
            # if delete_ignore: labelarr = np.delete(labelarr, ignore_cls)  # 有忽略像素时候需要用到
            label[image_name] = labelarr
            # print(image_name, labelarr)
    # print(label)
    np.save(save_path,label)


def npy_labels_to_txt(labelpath, savefile):
    data = np.load(labelpath, allow_pickle=True).item()
    with open(savefile, 'w') as f:
        for name, label in data.items():
            f.writelines(f'{name}\t{label}\n')


def main():
    pixel_label_to_tag(r'/media/ders/sdc1/FJM/datasets/Generality/I512s512/crop3/train/labels/',
                       r'/media/ders/sdc1/FJM/datasets/Generality/I512s512/crop3/train/classification/train_.npy',
                       16)  # 忽略像素备用
    npy_labels_to_txt(r'/media/ders/sdc1/FJM/datasets/Generality/I512s512/crop3/train/classification/train_.npy',
                      r'/media/ders/sdc1/FJM/datasets/Generality/I512s512/crop3/train/classification/train_.txt')


    return 0

if __name__=='__main__':
    main()


