import os
import cv2
import glob
import torch

import math
import imageio
import numpy as np

from PIL import Image

from core.aff_utils import *

from tools.ai.augment_utils import *
from tools.ai.torch_utils import one_hot_embedding

from tools.general.xml_utils import read_xml
from tools.general.json_utils import read_json
from tools.dataset.voc_utils import get_color_map_dic

class Iterator:
    def __init__(self, loader):
        self.loader = loader
        self.init()

    def init(self):
        self.iterator = iter(self.loader)
    
    def get(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.init()
            data = next(self.iterator)
        
        return data

class VOC_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, domain, with_id=False, with_tags=False, with_mask=False):
        self.root_dir = root_dir# 存储照片的文件夹目录

        # self.image_dir = self.root_dir + 'JPEGImages/'# 存照片路径
        # self.xml_dir = self.root_dir + 'Annotations/'# 存xml信息
        # self.mask_dir = self.root_dir + 'SegmentationClass/'# 存标签图片的文件夹
        if domain[0:5] == 'train':
            self.image_dir = self.root_dir + 'train/'+'images/'  # 存照片路径
            self.mask_dir = self.root_dir + 'train/'+'labels/'  # 存标签图片的文件夹
            self.label_npy = self.root_dir + 'train/'+'classification/'+'train_.npy'# 存标签图片的文件夹
            self.image_data_list = os.listdir(self.image_dir)# 带后缀的所有文件名
            self.image_id_list = [it[:-4] for it in self.image_data_list]# 不带后缀的所有文件名

        if domain[0:4] == 'val':
            self.image_dir = self.root_dir + 'val/'+'images/'  # 存照片路径
            self.mask_dir = self.root_dir + 'val/'+'labels/'# 存标签图片的文件夹
            self.label_npy = self.root_dir + 'val/'+'classification/'+'val_.npy'# 存标签图片的文件夹
            self.image_data_list = os.listdir(self.image_dir)# 带后缀的所有文件名
            self.image_id_list = [it[:-4] for it in self.image_data_list]# 不带后缀的所有文件名
        if domain[0:4] == 'test':
            self.image_dir = self.root_dir + 'test/'+'images/'  # 存照片路径
            self.mask_dir = self.root_dir + 'test/'+'labels/'# 存标签图片的文件夹
            # self.image_data_list = os.listdir(self.image_dir)# 带后缀的所有文件名
            # self.image_id_list = [it[:-4] for it in self.image_data_list]# 不带后缀的所有文件名

        # self.labels_dict = np.load(self.label_npy,allow_pickle=True).item()
        # self.images_name = [i for i in self.labels_dict]
        # self.images_dir = [os.path.join(self.image_dirs,image_name+'.tif') for image_name in self.labels_dict]
        # self.labels = list(self.labels_dict.values())
        # self.image_id_list = [image_id.strip() for image_id in open('./data/%s.txt'%domain).readlines()]  # strip()方法用于移除字符串开头和结尾的空格和换行符。
        
        self.with_id = with_id  # true
        self.with_tags = with_tags  # false
        self.with_mask = with_mask  # false

    def __len__(self):
        return len(self.image_id_list)

    def get_image(self, image_id):
        image = Image.open(self.image_dir + image_id + '.tif').convert('RGB')  # Image就是PIL那个包来读取图片，在文件夹中读取图片
        return image

    def get_mask(self, image_id):
        mask_path = self.mask_dir + image_id + '.png'
        if os.path.isfile(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = None
        return mask

    def get_tags(self, label_npy, image_id):
        # _, tags = read_xml(self.xml_dir + image_id + '.xml')
        self.labels_dict = np.load(label_npy, allow_pickle=True).item()
        tags = self.labels_dict.get(image_id, 'Key not found')
        # self.labels = list(self.labels_dict.values())
        # tags = torch.tensor(self.labels,dtype=float).view(-1)
        return tags
    
    def __getitem__(self, index):
        label_npy = self.label_npy
        image_id = self.image_id_list[index]  # txt文件中存放的文件名就不是带后缀的

        data_list = [self.get_image(image_id)]

        if self.with_id:
            data_list.append(image_id)

        if self.with_tags:
            data_list.append(self.get_tags(label_npy, image_id))

        if self.with_mask:
            data_list.append(self.get_mask(image_id))
        
        return data_list


class VOC_Datasetinferseg(torch.utils.data.Dataset):
    def __init__(self, root_dir, domain, with_id=False, with_tags=False, with_mask=False):
        self.root_dir = root_dir  # 存储照片的文件夹目录

        # self.image_dir = self.root_dir + 'JPEGImages/'# 存照片路径
        # self.xml_dir = self.root_dir + 'Annotations/'# 存xml信息
        # self.mask_dir = self.root_dir + 'SegmentationClass/'# 存标签图片的文件夹
        if domain[0:5] == 'train':
            self.image_dir = self.root_dir + 'train/' + 'images/'  # 存照片路径
            self.mask_dir = self.root_dir + 'train/' + 'labels/'  # 存标签图片的文件夹
            self.label_npy = self.root_dir + 'train/' + 'classificatiaon/' + 'train_.npy'  # 存标签图片的文件夹
            self.image_data_list = os.listdir(self.image_dir)  # 带后缀的所有文件名
            self.image_id_list = [it[:-4] for it in self.image_data_list]  # 不带后缀的所有文件名

        if domain[0:4] == 'val':
            self.image_dir = self.root_dir + 'val/' + 'images/'  # 存照片路径
            self.mask_dir = self.root_dir + 'val/' + 'labels/'  # 存标签图片的文件夹
            self.label_npy = self.root_dir + 'val/' + 'classificatiaon/' + 'val_.npy'  # 存标签图片的文件夹
            self.image_data_list = os.listdir(self.image_dir)  # 带后缀的所有文件名
            self.image_id_list = [it[:-4] for it in self.image_data_list]  # 不带后缀的所有文件名
        if domain[0:4] == 'test':
            self.image_dir = self.root_dir + 'test/' + 'images/'  # 存照片路径
            self.mask_dir = self.root_dir + 'test/' + 'labels/'  # 存标签图片的文件夹
            self.image_data_list = os.listdir(self.image_dir)  # 带后缀的所有文件名
            self.image_id_list = [it[:-4] for it in self.image_data_list]  # 不带后缀的所有文件名

        # self.labels_dict = np.load(self.label_npy,allow_pickle=True).item()
        # self.images_name = [i for i in self.labels_dict]
        # self.images_dir = [os.path.join(self.image_dirs,image_name+'.tif') for image_name in self.labels_dict]
        # self.labels = list(self.labels_dict.values())
        # self.image_id_list = [image_id.strip() for image_id in open('./data/%s.txt'%domain).readlines()]  # strip()方法用于移除字符串开头和结尾的空格和换行符。

        self.with_id = with_id  # true
        self.with_tags = with_tags  # false
        self.with_mask = with_mask  # false

    def __len__(self):
        return len(self.image_id_list)

    def get_image(self, image_id):
        image = Image.open(self.image_dir + image_id + '.tif').convert('RGB')  # Image就是PIL那个包来读取图片，在文件夹中读取图片
        return image

    def get_mask(self, image_id):
        mask_path = self.mask_dir + image_id + '.png'
        if os.path.isfile(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = None
        return mask

    def get_tags(self, label_npy, image_id):
        # _, tags = read_xml(self.xml_dir + image_id + '.xml')
        self.labels_dict = np.load(label_npy, allow_pickle=True).item()
        tags = self.labels_dict.get(image_id, 'Key not found')
        # self.labels = list(self.labels_dict.values())
        # tags = torch.tensor(self.labels,dtype=float).view(-1)
        return tags

    def __getitem__(self, index):
        # label_npy = self.label_npy
        image_id = self.image_id_list[index]  # txt文件中存放的文件名就不是带后缀的

        data_list = [self.get_image(image_id)]

        if self.with_id:
            data_list.append(image_id)

        # if self.with_tags:
        #     data_list.append(self.get_tags(label_npy, image_id))

        if self.with_mask:
            data_list.append(self.get_mask(image_id))

        return data_list



class VOC_Datasetseg(torch.utils.data.Dataset):
    def __init__(self, root_dir, domain, with_id=False, with_tags=False, with_mask=False):
        self.root_dir = root_dir  # 存储照片的文件夹目录

        # self.image_dir = self.root_dir + 'JPEGImages/'# 存照片路径
        # self.xml_dir = self.root_dir + 'Annotations/'# 存xml信息
        # self.mask_dir = self.root_dir + 'SegmentationClass/'# 存标签图片的文件夹
        if domain[0:5] == 'train':
            self.image_dir = self.root_dir + 'train/' + 'images/'  # 存照片路径
            self.mask_dir = self.root_dir + 'train/' + 'labels/'  # 存标签图片的文件夹
            # self.label_npy = self.root_dir + 'train/' + 'classificatiaon/' + 'train_.npy'  # 存标签图片的文件夹
            self.image_data_list = os.listdir(self.image_dir)  # 带后缀的所有文件名
            self.image_id_list = [it[:-4] for it in self.image_data_list]  # 不带后缀的所有文件名

        if domain[0:4] == 'val':
            self.image_dir = self.root_dir + 'val/' + 'images/'  # 存照片路径
            self.mask_dir = self.root_dir + 'val/' + 'labels/'  # 存标签图片的文件夹
            # self.label_npy = self.root_dir + 'val/' + 'classificatiaon/' + 'val_.npy'  # 存标签图片的文件夹
            self.image_data_list = os.listdir(self.image_dir)  # 带后缀的所有文件名
            self.image_id_list = [it[:-4] for it in self.image_data_list]  # 不带后缀的所有文件名
        if domain[0:4] == 'test':
            self.image_dir = self.root_dir + 'test/' + 'images/'  # 存照片路径
            self.mask_dir = self.root_dir + 'test/' + 'labels/'  # 存标签图片的文件夹
            # self.image_data_list = os.listdir(self.image_dir)# 带后缀的所有文件名
            # self.image_id_list = [it[:-4] for it in self.image_data_list]# 不带后缀的所有文件名

        # self.labels_dict = np.load(self.label_npy,allow_pickle=True).item()
        # self.images_name = [i for i in self.labels_dict]
        # self.images_dir = [os.path.join(self.image_dirs,image_name+'.tif') for image_name in self.labels_dict]
        # self.labels = list(self.labels_dict.values())
        # self.image_id_list = [image_id.strip() for image_id in open('./data/%s.txt'%domain).readlines()]  # strip()方法用于移除字符串开头和结尾的空格和换行符。

        self.with_id = with_id  # true
        self.with_tags = with_tags  # false
        self.with_mask = with_mask  # false

    def __len__(self):
        return len(self.image_id_list)

    def get_image(self, image_id):
        image = Image.open(self.image_dir + image_id + '.tif').convert('RGB')  # Image就是PIL那个包来读取图片，在文件夹中读取图片
        return image

    def get_mask(self, image_id):
        mask_path = self.mask_dir + image_id + '.png'
        if os.path.isfile(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = None
        return mask

    def get_tags(self, label_npy, image_id):
        # _, tags = read_xml(self.xml_dir + image_id + '.xml')
        self.labels_dict = np.load(label_npy, allow_pickle=True).item()
        tags = self.labels_dict.get(image_id, 'Key not found')
        # self.labels = list(self.labels_dict.values())
        # tags = torch.tensor(self.labels,dtype=float).view(-1)
        return tags

    def __getitem__(self, index):
        # label_npy = self.label_npy
        image_id = self.image_id_list[index]  # txt文件中存放的文件名就不是带后缀的

        data_list = [self.get_image(image_id)]

        if self.with_id:
            data_list.append(image_id)

        # if self.with_tags:
        #     data_list.append(self.get_tags(label_npy, image_id))

        if self.with_mask:
            data_list.append(self.get_mask(image_id))

        return data_list


class VOC_Dataset_For_Classification(VOC_Dataset):
    def __init__(self, root_dir, domain, transform=None):
        super().__init__(root_dir, domain, with_tags=True)
        self.transform = transform

        # data = read_json('./data/VOC_2012.json')

        # self.class_dic = data['class_dic']
        # self.classes = data['classes']

    def __getitem__(self, index):
        image, tags = super().__getitem__(index)

        if self.transform is not None:
            image = self.transform(image)
        label = tags
        # label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
        return image, label

class VOC_Dataset_For_Segmentation(VOC_Dataset):
    def __init__(self, root_dir, domain, transform=None):  # transform=None中的None表示空集
        super().__init__(root_dir, domain, with_mask=True)
        self.transform = transform

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])
    
    def __getitem__(self, index):
        image, mask = super().__getitem__(index)

        if self.transform is not None:
            input_dic = {'image':image, 'mask':mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']
            mask = output_dic['mask']
        
        return image, mask

class VOC_Dataset_For_Evaluation(VOC_Datasetinferseg):
    def __init__(self, root_dir, domain, transform=None):
        super().__init__(root_dir, domain, with_id=True, with_mask=True)
        self.transform = transform

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])

    def __getitem__(self, index):
        image, image_id, mask = super().__getitem__(index)

        if self.transform is not None:
            input_dic = {'image':image, 'mask':mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']
            mask = output_dic['mask']
        
        return image, image_id, mask

class VOC_Dataset_For_WSSS(VOC_Datasetseg):
    def __init__(self, root_dir, domain, pred_dir, transform=None):
        super().__init__(root_dir, domain, with_id=True)
        self.pred_dir = pred_dir# 黑白标签图片文件夹
        # self.weight_dir = weight_dir
        self.transform = transform

        cmap_dic, _, class_names = get_color_map_dic()# cmap_dic类别和颜色对照表；class_names类别名称
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])
    
    def __getitem__(self, index):
        image, image_id = super().__getitem__(index)
        mask = Image.open(self.pred_dir + image_id + '.png')
        # weight_image = cv2.imread(os.path.join(self.weight_dir, image_id + '.png'), cv2.IMREAD_UNCHANGED)

        # 如果你保存的是 8-bit 图像，需要将其还原为 [0, 1] 范围
        # weight = weight_image.astype(np.float32) / 255.0

        # weight = Image.open(self.weight_dir + image_id + '.png')
        
        if self.transform is not None:
            input_dic = {'image':image, 'mask':mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']# 经过transform的图片变为(3,512,512)
            mask = output_dic['mask']# mask是每一类对应的索引构成的1维图片(512,512)
        
        return image, mask

class VOC_Dataset_For_Testing_CAM(VOC_Dataset):
    def __init__(self, root_dir, domain, transform=None):
        super().__init__(root_dir, domain, with_tags=True, with_mask=True)
        self.transform = transform

        # cmap_dic, _, class_names = get_color_map_dic()
        # self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])
        #
        # data = read_json('./data/VOC_2012.json')
        #
        # self.class_dic = data['class_dic']
        # self.classes = data['classes']

    def __getitem__(self, index):
        image, tags, mask = super().__getitem__(index)

        if self.transform is not None:
            input_dic = {'image':image, 'mask':mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']
            mask = output_dic['mask']
        
        # label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
        label = tags
        return image, label, mask

class VOC_Dataset_For_Making_CAM(VOC_Dataset):
    def __init__(self, root_dir, domain):
        super().__init__(root_dir, domain, with_id=True, with_tags=True, with_mask=True)

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])
        
        data = read_json('./data/VOC_2012.json')

        self.class_names = np.asarray(class_names[1:6])
        self.class_dic = data['class_dic']
        self.classes = data['classes']

    def __getitem__(self, index):
        image, image_id, tags, mask = super().__getitem__(index)

        # label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
        label = tags
        return image, image_id, label, mask

class VOC_Dataset_For_Affinity(VOC_Dataset):
    def __init__(self, root_dir, domain, path_index, label_dir, transform=None):
        super().__init__(root_dir, domain, with_id=True)

        data = read_json('./data/VOC_2012.json')

        self.class_dic = data['class_dic']
        self.classes = data['classes']

        self.transform = transform

        self.label_dir = label_dir
        self.path_index = path_index

        self.extract_aff_lab_func = GetAffinityLabelFromIndices(self.path_index.src_indices, self.path_index.dst_indices)

    def __getitem__(self, idx):
        image, image_id = super().__getitem__(idx)

        label = imageio.imread(self.label_dir + image_id + '.png')
        label = Image.fromarray(label)
        
        output_dic = self.transform({'image':image, 'mask':label})
        image, label = output_dic['image'], output_dic['mask']
        
        return image, self.extract_aff_lab_func(label)

