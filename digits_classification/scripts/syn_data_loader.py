from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from PIL import Image

#warnings.filterwarnings("ignore")

class SYN_DATASET(Dataset):
	def __init__(self,csv_file=None,root=None,transform=None):
		self.root=root
		label_path=os.path.join(self.root,csv_file)

		self.label_frame=pd.read_csv(label_path,delimiter="\t")
		
		self.transform=transform

	def __len__(self):



		return len(self.label_frame)

	def __getitem__(self,idx):
		img_name="save_image"+str(idx)+".jpg"
		img_path=os.path.join(self.root,img_name)
		img=Image.open(img_path)
		labels=self.label_frame.iloc[idx,1]
		labels=np.array([labels])
		labels=labels.astype('int')
		if self.transform:
			img=self.transform(img)
		sample=(img,labels)
		return sample




if __name__ == '__main__':
	dataset=SYN_DATASET(csv_file='label.txt',root='generated_image/3t_123/1')
	print(dataset[0][1])












