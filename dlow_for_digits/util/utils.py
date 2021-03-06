import os
import json
import shutil
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.distributions.beta import Beta
import torch.nn.functional as F


def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def init_weights(network, init_type='normal', init_gain=0.02):
    def actual_init(model):
        classname = model.__class__.__name__
        if hasattr(model, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(model.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(model.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(model.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(model.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('Initialization method [%s] is not implemented' % init_type)

        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(model.weight.data, 1.0, init_gain)
            nn.init.constant_(model.bias.data, 0.0)
    
    network.apply(actual_init)


def get_scheduler(confimizer, args):
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + args.epoch_count - args.start_decay) / float(args.decay_cycle + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(confimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(confimizer, step_size=args.lr_decay_iters, gamma=0.1)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(confimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(confimizer, T_max=args.start_decay, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler


def tensor2img(tensor):

    img = ((tensor.cpu().float().numpy()) + 1) / 2.0 * 255.0
    if img.shape[0]==1:
        img = np.tile(img, (3, 1, 1))
    return img.astype(np.uint8)


def saveImg(tensor, output_dir, name): 

    img = tensor2img(tensor)
    save = img.copy()
    save = np.transpose(save, (1, 2, 0))
    save = Image.fromarray(save.astype('uint8'))
    save.save(os.path.join(output_dir, name))


def get_domainess(cur_iter, total_iter, batch):
    alpha = np.exp((cur_iter - (0.5 * total_iter)) / (0.25 * total_iter))
    distribution = Beta(alpha, 1)
    z=distribution.sample((batch, 1))
    output=torch.cat([z,1-z],dim=1)
    return output


def get_single_z(cur_iter, total_iter, batch):
    alpha = np.exp((cur_iter - (0.5 * total_iter)) / (0.25 * total_iter))
    distribution = Beta(alpha, 1)
    z=distribution.sample((batch, 1))
    return z




def get_random_domainess(cur_iter, total_iter, batch):
    alpha = np.exp((cur_iter - (0.5 * total_iter)) / (0.25 * total_iter))
    distribution = Beta(alpha, 1)
    z=distribution.sample((batch, 1))
    z2=z*torch.rand(1)
    output=torch.cat([1-z,z2,z-z2],dim=1)
    return output









def print_network(net, out_f=None):
    num_conf = 0
    for param in net.parameters():
        num_conf += param.numel()
    print(net)
    print("Total number of parameters: %d\n"%num_conf)
    if out_f is not None:
        out_f.write(net.__repr__()+"\n")
        out_f.write('Total number of parameters: %d\n' % num_conf)
        out_f.flush()