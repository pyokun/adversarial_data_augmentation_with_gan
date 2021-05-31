from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")




class TensorToDataset(Dataset):
	def __init__(self,x_set,y_set,transform=None):
		self.x_set=x_set
		self.y_set=y_set
		self.transform=transform

	def __len__(self):
		return len(self.x_set)

	def __getitem__(self,idx):
		if torch.is_tensor(idx):
			idx=idx.tolist()
		x=self.x_set[idx]
		y=self.y_set[idx]
		if self.transform:
			x=self.transform(x)
		return (x,y)

		













