# Copyright (C) 2021 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import torch.utils.model_zoo as model_zoo

from .arch_resnet import resnet
from .arch_resnest import resnest
from .abc_modules import ABC_Model

from .deeplab_utils import ASPP, Decoder
from .lanet_utils import Patch_Attention
from .aff_utils import PathIndex
from .puzzle_utils import tile_features, merge_features

from tools.ai.torch_utils import resize_for_tensors

#######################################################################
# Normalization
#######################################################################
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

def group_norm(features):
    return nn.GroupNorm(4, features)
#######################################################################

class Backbone(nn.Module, ABC_Model):
    def __init__(self, model_name, num_classes=20, mode='fix', segmentation=False):
        super().__init__()

        self.mode = mode

        if self.mode == 'fix': 
            self.norm_fn = FixedBatchNorm
        else:
            self.norm_fn = nn.BatchNorm2d
        
        if 'resnet' in model_name:
            self.model = resnet.ResNet(resnet.Bottleneck, resnet.layers_dic[model_name], strides=(2, 2, 2, 1), batch_norm_fn=self.norm_fn)
            # self.model = resnet.ResNet(resnet.BasicBlock, resnet.layers_dic[model_name], strides=(2, 2, 2, 1),
            #                            batch_norm_fn=self.norm_fn)  # Basicneck是34所用
            # state_dict = model_zoo.load_url(resnet.urls_dic[model_name])
            # state_dict = torch.load("/media/ders/Fjmnew/RSIPuzzleCAM/pretrained/resnet34-333f7ec4.pth")
            state_dict = torch.load("/media/ders/sdc1/FJM/115P4mul_ipc_AP_aff/pretrained/resnet50-19c8e357.pth")
            state_dict.pop('fc.weight')  # 弹出全连接的权重，可以理解为在字典中删除全连接层的权重
            state_dict.pop('fc.bias')  # 弹出全连接的偏置

            self.model.load_state_dict(state_dict)
        else:
            if segmentation:
                dilation, dilated = 4, True
            else:
                dilation, dilated = 2, False

            self.model = eval("resnest." + model_name)(pretrained=True, dilated=dilated, dilation=dilation, norm_layer=self.norm_fn)

            del self.model.avgpool
            del self.model.fc

        self.stage1 = nn.Sequential(self.model.conv1,
                                    self.model.bn1,
                                    self.model.relu,
                                    self.model.maxpool)
        # self.co = self.model.conv1  # unetCAM用
        # self.bn = self.model.bn1
        # self.re = self.model.relu
        # self.po = self.model.maxpool
        self.stage2 = nn.Sequential(self.model.layer1)
        self.stage3 = nn.Sequential(self.model.layer2)
        self.stage4 = nn.Sequential(self.model.layer3)
        self.stage5 = nn.Sequential(self.model.layer4)

class Classifier(Backbone):
    def __init__(self, model_name, num_classes=20, mode='fix'):
        super().__init__(model_name, num_classes, mode)
        self.side1 = nn.Conv2d(256, 128, 1, bias=False)
        self.side2 = nn.Conv2d(512, 128, 1, bias=False)
        self.side3 = nn.Conv2d(1024, 256, 1, bias=False)
        self.side4 = nn.Conv2d(2048, 256, 1, bias=False)

        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
        self.f9 = torch.nn.Conv2d(3 + 2048, 2048, 1, bias=False)
        self.num_classes = num_classes
        self.initialize([self.classifier])  # 初始化分类层的权重

    def PCM(self, cam, f, pcm, th):
        n,c,h,w = f.size()
        cam = F.interpolate(cam, (h,w), mode='bilinear', align_corners=True).view(n,-1,h*w)  # 8 6 64
        f = self.f9(f)  # 8 2048 8 8
        f = f.view(n,-1,h*w)  # 8 2048 64
        f = F.normalize(f, dim=1)  # 相当于余弦相似性，归一化后模为1，便于计算，技巧
        aff_b = torch.bmm(f.transpose(1, 2), f)  # 64 64将 x4.transpose(1, 2) =64*1024与 x4 进行矩阵乘法1024*64=64*64
        aff = torch.clamp(aff_b, 0.01, 0.999)  # 8 64 64将 input 张量中的所有元素限制在指定的范围 [min, max] 之间
        aff[aff < th] = 0  # gama选择
        aff = aff/aff.sum(1, True)  # 归一化8 64 64
        logits_flat = cam.view(n, self.num_classes, -1)  # 8 6 64
        for i in range(pcm):
            logits_flat = torch.bmm(logits_flat, aff)  # 随机游走策略
        cam_rv = logits_flat.view(n, self.num_classes, h, w)  # 提出的CAM

        # f = f/(torch.norm(f,dim=1,keepdim=True)+1e-5)
        # aff = F.relu(torch.matmul(f.transpose(1,2), f),inplace=True)  # 像素极关系
        # aff = aff/(torch.sum(aff,dim=1,keepdim=True)+1e-5)  # 关系矩阵
        # cam_rv = torch.matmul(cam, aff).view(n,-1,h,w)  # 8 6 8 8
        # # cam_rv = torch.matmul(cam_rv.view(n,-1,h*w), aff).view(n,-1,h,w)
        return cam_rv


    def prototype(self, norm_cam, feature):
        n,c,h,w = norm_cam.shape
        # norm_cam[:,0] = norm_cam[:,0]*0.3
        seeds = torch.zeros((n,h,w,c)).cuda()
        belonging = norm_cam.argmax(1)
        seeds = seeds.scatter_(-1, belonging.view(n,h,w,1), 1).permute(0,3,1,2).contiguous()
        # seeds = seeds * valid_mask # 4, 21, 32, 32

        n,c,h,w = feature.shape # hie
        seeds = F.interpolate(seeds, feature.shape[2:], mode='nearest')
        crop_feature = seeds.unsqueeze(2) * feature.unsqueeze(1) #.clone().detach()  # seed:[n,21,1,h,w], feature:[n,1,4c,h,w], crop_feature:[n,21,4c,h,w]
        prototype = F.adaptive_avg_pool2d(crop_feature.view(-1,c,h,w), (1,1)).view(n, self.num_classes, c, 1, 1) # prototypes:[n,21,c,1,1]

        IS_cam = F.relu(torch.cosine_similarity(feature.unsqueeze(1), prototype, dim=2)) # feature:[n,1,4c,h,w], prototypes:[n,21,4c,1,1], crop_feature:[n,21,h,w]
        IS_cam = F.interpolate(IS_cam, feature.shape[2:], mode='bilinear', align_corners=True)
        return IS_cam

    def forward(self, x, with_cam=False, pcm=1, th=0.7):
        x0 = self.stage1(x)  # 8 64 32 32
        x1 = self.stage2(x0)
        x2 = self.stage3(x1).detach()
        x3 = self.stage4(x2)
        x4 = self.stage5(x3)
        side1 = self.side1(x1.detach())
        side2 = self.side2(x2.detach())
        side3 = self.side3(x3.detach())
        side4 = self.side4(x4.detach())
        # print(self.side1.weight)
        sem_feature = x4  # 8 2048 8 8
        cam = self.classifier(x4)
        logits = self.global_average_pooling_2d(self.classifier(x4))

        # norm_cam = F.relu(cam)
        # norm_cam = norm_cam/(F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        # # cam_bkg = 1-torch.max(norm_cam,dim=1)[0].unsqueeze(1)
        # # norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        # norm_cam = F.interpolate(norm_cam, side3.shape[2:], mode='bilinear', align_corners=True)
        # orignal_cam = norm_cam

        hie_fea = torch.cat([F.interpolate(side1/(torch.norm(side1,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'),
                            F.interpolate(side2/(torch.norm(side2,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'),
                            F.interpolate(side3/(torch.norm(side3,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear'),
                            F.interpolate(side4/(torch.norm(side4,dim=1,keepdim=True)+1e-5), side3.shape[2:], mode='bilinear')], dim=1)
        fusion_cam = torch.cat([F.interpolate(x, side3.shape[2:],mode='bilinear',align_corners=True), sem_feature], dim=1)  #8 2051 8 8
        norm_cam = self.PCM(cam, fusion_cam, pcm, th)
        # IS_cam = self.prototype(norm_cam.clone(), hie_fea.clone())

        return logits, norm_cam
        # if with_cam:  # 在预测时候用到生成CAM
        #     features = self.classifier(x5)  # 8 6 8 8
        #     logits = self.global_average_pooling_2d(features)
        #     return logits, features

        # if with_cam:  # 在预测时候用到生成CAM
        #     features = self.classifier(x4)  # 8 6 8 8
        #     logits = self.global_average_pooling_2d(self.classifier(x4))  # 原始分类网络的分数8 6
        #     # AP模块
        #     x4 = torch.cat([x4], dim=1)  # 8 1024 8 8
        #     b, c, h, w = x4.shape
        #     x4 = x4.view(b, c, -1)  # 8 1024 64     256=16*16 把16*16的特征chen直了变为256，三维变为二维矩阵1024行256列
        #     x4 = F.normalize(x4, dim=1)  # 相当于余弦相似性，归一化后模为1，便于计算，技巧
        #     aff_b = torch.bmm(x4.transpose(1, 2), x4)  # 64 64将 x4.transpose(1, 2) =64*1024与 x4 进行矩阵乘法1024*64=64*64
        #     aff = torch.clamp(aff_b, 0.01, 0.999)  # 8 64 64将 input 张量中的所有元素限制在指定的范围 [min, max] 之间
        #     aff[aff < th] = 0  # gama选择
        #     aff = aff/aff.sum(1, True)  # 归一化8 64 64
        #     logits_flat = features.view(b, self.num_classes, -1)  # 8 6 64
        #     for i in range(pcm):
        #         logits_flat = torch.bmm(logits_flat, aff)  # 随机游走策略
        #     features = logits_flat.view(b, self.num_classes, h, w)  # 提出的CAM
        #     IS_cam = self.prototype(norm_cam.clone(), hie_fea.clone(), valid_mask.clone())
        #
        #     # ############语义特征语义亲和力加权#############
        #     # sem_feature = x5  # 语义特征
        #     # sem_feature = torch.cat([sem_feature], dim=1)  # 8 1024 8 8
        #     # b5, c5, h5, w5 = x5.shape
        #     # sem_feature = sem_feature.view(b5, c5, -1)  # 8 2048 64     256=16*16 把16*16的特征chen直了变为256，三维变为二维矩阵1024行256列
        #     # sem_feature = F.normalize(sem_feature, dim=1)  # 相当于余弦相似性，归一化后模为1，便于计算，技巧
        #     # aff_b5 = torch.bmm(sem_feature.transpose(1, 2), sem_feature)  # 64 64将 x4.transpose(1, 2) =64*1024与 x4 进行矩阵乘法1024*64=64*64
        #     # aff5 = torch.clamp(aff_b5, 0.01, 0.999)  # 8 64 64将 input 张量中的所有元素限制在指定的范围 [min, max] 之间
        #     # aff5[aff5 < 0.8] = 0  # gama选择
        #     # aff5 = aff5/aff5.sum(1, True)  # 归一化8 64 64 得到了语义关系矩阵
        #     # cam_flat = features.view(b5, self.num_classes, -1)  # 8 6 64
        #     # cam_flat = torch.bmm(cam_flat, aff5)
        #     # cam_final = cam_flat.view(b5, self.num_classes, h5, w5)
        #     return logits, cam_final
        # else:
        #     x = self.global_average_pooling_2d(x5, keepdims=True)   # 在训练过程中需要分类层的权重 8 2048 1 1
        #     logits = self.classifier(x).view(-1, self.num_classes)  # 输出对应的预测概率
        #     return logits

class CLSNet(Backbone):
    def __init__(self, model_name, num_classes=20, mode='fix'):
        super().__init__(model_name, num_classes, mode)

        self.num_classes =num_classes
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
        self.initialize([self.classifier])  # 初始化分类层的权重

    def forward(self, inputs, pcm=12, th=0.6):
        x = self.stage1(inputs)
        x = self.stage2(x)
        x = self.stage3(x).detach()
        x4 = self.stage4(x)
        x5 = self.stage5(x4)
        logits = self.classifier(x5)  # logits就是feature，即CAM
        logits_min =(F.adaptive_avg_pool2d(self.classifier(x5), 1))  # 正常的分类分数

        # AP模块
        if (pcm > 0):
            x4 = torch.cat([x4], dim=1)
            b, c, h, w = x4.shape
            x4 = x4.view(b, c, -1)
            x4 = F.normalize(x4, dim=1)  # 堆叠特征图，然后归一化
            aff_b = torch.bmm(x4.transpose(1, 2), x4)  # 将 x4.transpose(1, 2) 与 x4 进行矩阵乘法
            aff = torch.clamp(aff_b, 0.01, 0.999)  # 将 input 张量中的所有元素限制在指定的范围 [min, max] 之间
            # aff[aff < th] = 0  # gama选择
            aff = aff/aff.sum(1, True)
            logits_flat = logits.view(b, self.num_classes, -1)
            for i in range(pcm):
                logits_flat = torch.bmm(logits_flat, aff)  # 相乘操作
            logits = logits_flat.view(b, self.num_classes, h, w)  # 提出的CAM

        return logits, logits_min, x4









class Classifier_For_Positive_Pooling(Backbone):
    def __init__(self, model_name, num_classes=20, mode='fix'):
        super().__init__(model_name, num_classes, mode)
        
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
        self.num_classes = num_classes
        
        self.initialize([self.classifier])
    
    def forward(self, x, with_cam=False):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        
        if with_cam:
            features = self.classifier(x)
            logits = self.global_average_pooling_2d(features)
            return logits, features
        else:
            x = self.global_average_pooling_2d(x, keepdims=True) 
            logits = self.classifier(x).view(-1, self.num_classes)
            return logits

class Classifier_For_Puzzle(Classifier):
    def __init__(self, model_name, num_classes=20, mode='fix'):
        super().__init__(model_name, num_classes, mode)
        
    def forward(self, x, num_pieces=1, level=-1):
        batch_size = x.size()[0]
        
        output_dic = {}
        layers = [self.stage1, self.stage2, self.stage3, self.stage4, self.stage5, self.classifier]

        for l, layer in enumerate(layers):
            l += 1
            if level == l:
                x = tile_features(x, num_pieces)

            x = layer(x)
            output_dic['stage%d'%l] = x
        
        output_dic['logits'] = self.global_average_pooling_2d(output_dic['stage6'])

        for l in range(len(layers)):
            l += 1
            if l >= level:
                output_dic['stage%d'%l] = merge_features(output_dic['stage%d'%l], num_pieces, batch_size)

        if level is not None:
            output_dic['merged_logits'] = self.global_average_pooling_2d(output_dic['stage6'])

        return output_dic
        
class AffinityNet(Backbone):
    def __init__(self, model_name, path_index=None):
        super().__init__(model_name, None, 'fix')

        if '50' in model_name:
            fc_edge1_features = 64
        else:
            fc_edge1_features = 128

        self.fc_edge1 = nn.Sequential(
            nn.Conv2d(fc_edge1_features, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge2 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge3 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge4 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge5 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge6 = nn.Conv2d(160, 1, 1, bias=True)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
        self.edge_layers = nn.ModuleList([self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6])

        if path_index is not None:
            self.path_index = path_index
            self.n_path_lengths = len(self.path_index.path_indices)
            for i, pi in enumerate(self.path_index.path_indices):
                self.register_buffer("path_indices_" + str(i), torch.from_numpy(pi))
    
    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()

    def forward(self, x, with_affinity=False):
        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3).detach()
        x5 = self.stage5(x4).detach()
        
        edge1 = self.fc_edge1(x1)
        edge2 = self.fc_edge2(x2)
        edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
        edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
        edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]

        edge = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))  # 16 1 64 64

        if with_affinity:
            return edge, self.to_affinity(torch.sigmoid(edge))
        else:
            return edge

    def get_edge(self, x, image_size=512, stride=4):
        feat_size = (x.size(2)-1)//stride+1, (x.size(3)-1)//stride+1

        x = F.pad(x, [0, image_size-x.size(3), 0, image_size-x.size(2)])
        edge_out = self.forward(x)
        edge_out = edge_out[..., :feat_size[0], :feat_size[1]]
        edge_out = torch.sigmoid(edge_out[0]/2 + edge_out[1].flip(-1)/2)
        
        return edge_out
    
    """
    aff = self.to_affinity(torch.sigmoid(edge_out))
    pos_aff_loss = (-1) * torch.log(aff + 1e-5)
    neg_aff_loss = (-1) * torch.log(1. + 1e-5 - aff)
    """
    def to_affinity(self, edge):
        aff_list = []
        edge = edge.view(edge.size(0), -1)  # 转换为 16  4096=64*64 16个batch总共有4096个像素
        
        for i in range(self.n_path_lengths):
            ind = self._buffers["path_indices_" + str(i)]
            ind_flat = ind.view(-1)
            dist = torch.index_select(edge, dim=-1, index=ind_flat)
            dist = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2))
            aff = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2)
            aff_list.append(aff)
        aff_cat = torch.cat(aff_list, dim=1)
        return aff_cat


class IDeepLabv3(Backbone):
    def __init__(self, model_name, num_classes=21, mode='fix', use_group_norm=False):
        super().__init__(model_name, num_classes, mode, segmentation=False)

        if use_group_norm:
            norm_fn_for_extra_modules = group_norm
        else:
            norm_fn_for_extra_modules = self.norm_fn

        self.aspp = ASPP(output_stride=16, norm_fn=norm_fn_for_extra_modules)
        self.decoder = Decoder(num_classes, 256, norm_fn_for_extra_modules)
        self.PA2 = Patch_Attention(256, reduction=16, pool_window=4, add_input=True)

    def forward(self, x, with_cam=False):
        inputs = x  # (8,3,512,512)

        x = self.stage1(x)  # (8,64,128,128)
        x = self.stage2(x)  # (8,256,128,128)
        x_low_level = x  # Resnet50的block2作为低阶特征

        x = self.stage3(x)  # (8,512,64,64)
        x = self.stage4(x)  # (8,1024,32,32)
        x = self.stage5(x)  # Resnet完事(8,2048,32,32)

        x = self.aspp(x)  # 4  256 32 32
        x = self.PA2(x)  # (4,256,32,32)
        x = self.decoder(x, x_low_level)  # (8,21,128,128)
        x = resize_for_tensors(x, inputs.size()[2:],
                               align_corners=True)  # (4,6,512,512)align_corners=True在两个像素点之间进行插值，点之间是等距的
        # x = self.nunet(x)

        return x


class DeepLabv3_Plus(Backbone):
    def __init__(self, model_name, num_classes=21, mode='fix', use_group_norm=False):
        super().__init__(model_name, num_classes, mode, segmentation=False)
        
        if use_group_norm:
            norm_fn_for_extra_modules = group_norm
        else:
            norm_fn_for_extra_modules = self.norm_fn
        
        self.aspp = ASPP(output_stride=16, norm_fn=norm_fn_for_extra_modules)
        self.decoder = Decoder(num_classes, 256, norm_fn_for_extra_modules)
        
    def forward(self, x, with_cam=False):
        inputs = x# (8,3,512,512)

        x = self.stage1(x)# (8,64,128,128)
        x = self.stage2(x)# (8,256,128,128)
        x_low_level = x# Resnet50的block2作为低阶特征
        
        x = self.stage3(x)# (8,512,64,64)
        x = self.stage4(x)# (8,1024,32,32)
        x = self.stage5(x)# Resnet完事(8,2048,32,32)
        
        x = self.aspp(x)
        x = self.decoder(x, x_low_level)# (8,21,128,128)
        x = resize_for_tensors(x, inputs.size()[2:], align_corners=True)# (8,21,512,512)align_corners=True在两个像素点之间进行插值，点之间是等距的

        return x

class Seg_Model(Backbone):
    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, mode='fix', segmentation=False)
        
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)
    
    def forward(self, inputs):
        x = self.stage1(inputs)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        
        logits = self.classifier(x)
        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        
        return logits

class CSeg_Model(Backbone):
    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, 'fix')

        if '50' in model_name:
            fc_edge1_features = 64
        else:
            fc_edge1_features = 128

        self.fc_edge1 = nn.Sequential(
            nn.Conv2d(fc_edge1_features, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge2 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge3 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge4 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge5 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge6 = nn.Conv2d(160, num_classes, 1, bias=True)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        
        edge1 = self.fc_edge1(x1)
        edge2 = self.fc_edge2(x2)
        edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
        edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
        edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]

        logits = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))
        # logits = resize_for_tensors(logits, x.size()[2:], align_corners=True)
        
        return logits


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
    # def __init__(self):
    #     super(VGGBlock).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class UnetCAM(Backbone):
    def __init__(self, model_name, num_classes=20, mode='fix'):
        super().__init__(model_name, num_classes, mode)

        self.classifier = nn.Conv2d(64, num_classes, 1, bias=False)
        self.num_classes = num_classes

        self.initialize([self.classifier])  # 初始化分类层的权重
        # self.initialize([self.conv1])
        nb_filter = [64, 256, 512, 1024, 2048]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv3_4 = VGGBlock(nb_filter[4]+nb_filter[3], nb_filter[3], nb_filter[3])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_2 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_3 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
    def forward(self, x, with_cam=False):
        inputsfirst = self.co(x)
        x = self.bn(inputsfirst)  # 4 64 256 256
        x = self.re(x)
        x = self.po(x)
        x = self.stage2(x)  # (4,256,128,128)
        e1 = x
        x = self.stage3(x)  # (8,512,64,64)
        e2 = x
        x = self.stage4(x)  # (8,1024,32,32)
        e3 = x
        x = self.stage5(x)
        e4 = x
        x0_0 = inputsfirst
        x1_0 = e1
        x2_0 = e2
        x3_0 = e3
        x4_0 = e4
        x3_4 = self.conv3_4(torch.cat([x3_0, x4_0], 1))  # 8 512 32 32
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_4)], 1))  # 8 512 32 32
        x1_2 = self.conv1_2(torch.cat([x1_0, self.up(x2_1)], 1))  # 8 256 64 64
        output = self.conv0_3(torch.cat([x0_0, self.up(x1_2)], 1))  # 8 64 128 128
        # output = self.final(self.up(x0_3))

        if with_cam:  # 在预测时候用到生成CAM
            features = self.classifier(output)
            logits = self.global_average_pooling_2d(features)
            return logits, features
        else:
            output = self.global_average_pooling_2d(output, keepdims=True)  # 在训练过程中需要分类层的权重 8 2048 1 1
            logits = self.classifier(output).view(-1, self.num_classes)  # 输出对应的预测概率
            return logits