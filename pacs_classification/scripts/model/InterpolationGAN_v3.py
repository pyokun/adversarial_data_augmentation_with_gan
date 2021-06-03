import os
import sys
import itertools

import torch
import torch.nn as nn
from torch.autograd import Variable

from model import BaseNetwork as Base
from util import utils


from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR




sys.path.append("..")

class InterpolationGAN(nn.Module):

    def __init__(self, conf, device, is_train=True):
        super(InterpolationGAN, self).__init__()
        self.conf = conf
        self.device = device

        # Generator 
        self.G_ST = Base.Stoch_Generator(24, 3, 3, conf.g_conv_dim, 
                                        False, conf.n_res_blocks)                 
                         

    def forward(self, x, d):
        self.G_ST.eval()
        Z = d.view(d.size(0),3,1,1).repeat(1,8,1,1)                       # domainess Z (1, 16, 1, 1)
        fake_T = self.G_ST(x,Z)                                    # 
        return fake_T
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.G_ST.load_state_dict(checkpoint['G_ST'])
