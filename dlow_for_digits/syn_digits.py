import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
import numpy as np
import scipy.io as sio
import gzip
import wget
import h5py
import pickle
import urllib
import os
import skimage
import skimage.transform
from skimage.io import imread
import matplotlib.image as mpimg
import torch.nn.functional as F

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors


    def __getitem__(self, index):
        x = self.tensors[0][index]

        
        x=(x/255.0)*2-1.0
        x = F.interpolate(x,32)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)




def Loadsyn(data_root='syn_digits/',split='train',shuffle=True):
    if split=='train':
        mat=sio.loadmat(data_root+'synth_train_32x32.mat')
    elif split=='test':
        mat=sio.loadmat(data_root+'synth_test_32x32.mat')

    data=mat['X']
    target=mat['y']


    data=torch.from_numpy(data).float()
    target=torch.from_numpy(target).long().squeeze(1)
    data=data.permute(3,2,0,1)

    return_dataset=CustomTensorDataset(tensors=(data, target))
    return return_dataset


