import math

import torch
import torch.nn.functional as F

def tile_features(features, num_pieces):
    _, _, h, w = features.size()

    num_pieces_per_line = int(math.sqrt(num_pieces))  # 用于确定在一个正方形或矩形中，每行或每列可以容纳多少个相同大小的小块或物品。
    
    h_per_patch = h // num_pieces_per_line  # 分块高
    w_per_patch = w // num_pieces_per_line  # 分块宽
    
    """
    +-----+-----+
    |  1  |  2  |
    +-----+-----+
    |  3  |  4  |
    +-----+-----+

    +-----+-----+-----+-----+
    |  1  |  2  |  3  |  4  |
    +-----+-----+-----+-----+
    """
    patches = []
    for splitted_features in torch.split(features, h_per_patch, dim=2):  # 这行代码是使用PyTorch中的split函数，对features进行切分操作。参数h_per_patch指定了每个切分的大小，而dim=2表示在第二个维度上进行切分。
        for patch in torch.split(splitted_features, w_per_patch, dim=3):
            patches.append(patch)
    
    return torch.cat(patches, dim=0)

def merge_features(features, num_pieces, batch_size):
    """
    +-----+-----+-----+-----+
    |  1  |  2  |  3  |  4  |
    +-----+-----+-----+-----+
    
    +-----+-----+
    |  1  |  2  |
    +-----+-----+
    |  3  |  4  |
    +-----+-----+
    """
    features_list = list(torch.split(features, batch_size))  # list是将图片列成一片，32 20 16 16按照batch将这一组特征图平均分成4份特征图
    num_pieces_per_line = int(math.sqrt(num_pieces))  # math是python中的函数包，.sqrt就是平方根
    
    index = 0
    ext_h_list = []

    for _ in range(num_pieces_per_line):  # 现在第一行的方向上进行拼接接着第二行

        ext_w_list = []  # 临时存储分块图片
        for _ in range(num_pieces_per_line):
            ext_w_list.append(features_list[index])  # append 表示在宽度上 行上边一个一个添加的ext_w_list中
            index += 1
        
        ext_h_list.append(torch.cat(ext_w_list, dim=3))  # 第一行上 存储完两个块后，按行将两个特征块进行拼接

    features = torch.cat(ext_h_list, dim=2)  # 上边两个块下边两个块
    return features

def puzzle_module(x, func_list, num_pieces):
    tiled_x = tile_features(x, num_pieces)

    for func in func_list:
        tiled_x = func(tiled_x)
        
    merged_x = merge_features(tiled_x, num_pieces, x.size()[0])
    return merged_x
