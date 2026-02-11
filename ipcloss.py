


from weakref import ref
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset

from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *
from tools.general.Q_util import *
from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
from torch.nn.modules.loss import _Loss



def pool_feat_2(probs, feats):

    """
    probs:cam图
    feats：监督信息

    """
    b, cp, h, w = probs.shape  # 32 6 4 4
    b, cf, h, w = feats.shape  # 32 3 4 4 原图监督信息
    probs = probs.view(b, cp, h*w).transpose(1, 2)  # 32 16 6
    probs_sum = torch.sum(probs, dim=1, keepdim=True)  # 32 1 6
    feats = feats.view(b, cf, h*w)  # 32 3 16 展平操作
    ret = torch.bmm(feats, probs) / (probs_sum + 1e-5)  # 32 3 6

    return ret

def up_feat_2(probs, feats_min):
    b, cp, h, w = probs.shape
    b, cf, cp = feats_min.shape  # 32 3 6
    probs = probs.view(b, cp, h*w)   # 32 6 16
    ret = torch.bmm(feats_min, probs).view(b, cf, h, w)  #32 3 4 4

    return ret



class SP_CAM_Loss2(_Loss):
  def __init__(self, args, size_average=None, reduce=None, reduction='mean'):
    super(SP_CAM_Loss2, self).__init__(size_average, reduce, reduction)
    self.args = args
    self.fg_c_num = 6
    self.class_loss_fn = nn.CrossEntropyLoss().cuda()

  def forward(self, fg_cam, sailencys):#sailencys.max()

        """

        fg_cam：cam的切片
        saliencys：下采样的RGB图，监督信息的切片，并不是原来大小的图片

        """

        b, c, h, w = fg_cam.size()  # 32 6 128 128
        imgmin_mask = sailencys  # 32 3 128 128

        # imgmin_mask = sailencys.sum(1, True)  # 32 1 128 128对RGB三个通道每一个像素点进行求和构成，单通道patch图
        # origin_f = F.sigmoid(imgmin_mask)
        # sailencys = F.interpolate(sailencys.float(), size=(h, w))   # 32 3 8 8原图变为和CAM一样的大小，三通道patch图

        # bg = 1-torch.max(fg_cam, dim=1, keepdim=True)[0] ** 1
        #
        # nnn = torch.max((1 - bg.detach() * imgmin_mask).view(b, 1, -1), dim=2)[0] > self.args.ig_th
        # nnn2 = torch.max((bg.detach() * imgmin_mask).view(b, 1, -1), dim=2)[0] > self.args.ig_th
        # nnn = nnn * nnn2
        # if (nnn.sum() == 0):
        #   nnn = torch.ones(nnn.shape).cuda()
        # imgmin_mask = nnn.view(b, 1, 1, 1) * imgmin_mask    # 利用这个使CAM和其他的原图保持相同的维度

        # probs = torch.cat([bg, fg_cam], dim=1)  # 预测的cam和背景融合了
        probs1 = fg_cam  # 32 6 128 128
        # probs1 = probs * imgmin_mask

        origin_f = F.normalize(sailencys.detach(), dim=1)  # 对原图进行归一化 3 128 128
        origin_f = origin_f.double()  # 32 3 8 8
        # origin_f = origin_f
        # origin_f = origin_f * imgmin_mask

        f_min = pool_feat_2(probs1, origin_f)  # 公式6 ----32 3 * 6
        up_f = up_feat_2(probs1, f_min)  # 公式5 ----32 3 128 128

        sal_loss = F.mse_loss(up_f, origin_f, reduce=False)   # 均方误差（Mean Squared Error, MSE）损失函数
        # sal_loss = torch.sqrt(torch.sum(sal_loss, dim=1))  # 形状 (32, 4, 4)
        # sal_loss = (sal_loss * imgmin_mask).sum() / (torch.sum(imgmin_mask) + 1e-3)

        return sal_loss

