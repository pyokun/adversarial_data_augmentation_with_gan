import sys
import numpy as np

from tensorboardX import SummaryWriter

from . import utils

class Logger():
    def __init__(self, conf):
        self.conf = conf
        self.summary = SummaryWriter()

    def log(self, losses=None, images=None):

        for img_name, tensor in images.items(): 
            img = ((tensor.detach().cpu().float().numpy()) + 1) / 2.0
            self.summary.add_image(img_name, img, self.conf['cur_iter'])
            
        for loss_name, loss in losses.items():
            self.summary.add_scalar(loss_name, loss, self.conf['cur_iter'])