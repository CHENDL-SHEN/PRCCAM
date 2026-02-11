


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
    b, cp, h, w = probs.shape    
    b, cf, h, w = feats.shape   
    probs = probs.view(b, cp, h*w).transpose(1, 2)     
    probs_sum = torch.sum(probs, dim=1, keepdim=True)   
    feats = feats.view(b, cf, h*w)   
    ret = torch.bmm(feats, probs) / (probs_sum + 1e-5)  

    return ret

def up_feat_2(probs, feats_min):
    b, cp, h, w = probs.shape    
    b, cf, cp = feats_min.shape  
    probs = probs.view(b, cp, h*w)      
    ret = torch.bmm(feats_min, probs).view(b, cf, h, w)  

    return ret



class SP_CAM_Loss2(_Loss):
  def __init__(self, args, size_average=None, reduce=None, reduction='mean'):
    super(SP_CAM_Loss2, self).__init__(size_average, reduce, reduction)
    self.args = args
    self.fg_c_num = 20 if args.dataset == 'voc12' else 80
    self.class_loss_fn = nn.CrossEntropyLoss().cuda()

  def forward(self, fg_cam, sailencys):#sailencys.max()

        """

        fg_cam：cam
        saliencys：下采样的RGB图，监督信息

        """

        b, c, h, w = fg_cam.size()                 
        imgmin_mask = sailencys.sum(1, True) != 0   
        sailencys = F.interpolate(sailencys.float(), size=(h, w))   

        bg = 1-torch.max(fg_cam, dim=1, keepdim=True)[0] ** 1       

        nnn = torch.max((1 - bg.detach() * imgmin_mask).view(b, 1, -1), dim=2)[0] > self.args.ig_th  
        nnn2 = torch.max((bg.detach() * imgmin_mask).view(b, 1, -1), dim=2)[0] > self.args.ig_th      
        nnn = nnn * nnn2        
        if (nnn.sum() == 0):
          nnn = torch.ones(nnn.shape).cuda()
        imgmin_mask = nnn.view(b, 1, 1, 1) * imgmin_mask    

        probs = torch.cat([bg, fg_cam], dim=1)             
        probs1 = probs * imgmin_mask                       

        origin_f = F.normalize(sailencys.detach(), dim=1)  
        origin_f = origin_f * imgmin_mask                  

        f_min = pool_feat_2(probs1, origin_f)              
        up_f = up_feat_2(probs1, f_min)                   

        sal_loss = F.mse_loss(up_f, origin_f, reduce=False)          
        sal_loss = (sal_loss * imgmin_mask).sum() / (torch.sum(imgmin_mask) + 1e-3) 

        return sal_loss

