import PIL
import torch
import torch.nn as nn

from mnist_m import MNISTM
from torchvision import transforms

from syn_digits import Loadsyn
from torchvision.utils import save_image
from model.InterpolationGAN_v3 import InterpolationGAN

from domain_style_transfer import minibatch_transfer
import numpy as np
import torchvision
import parser    
import argparse
parser = argparse.ArgumentParser(description='Domain generalization')



parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--g_conv_dim', type=int, default=64)
parser.add_argument('--d_conv_dim', type=int, default=64)

parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--lambda_cycle',type=float,default=10.0)
parser.add_argument('--n_res_blocks',type=int,default=3)
args= parser.parse_args()





device = torch.device('cuda')
trans = transforms.Compose([transforms.Resize(size=[32, 32]),transforms.ToTensor() ,transforms.Lambda(lambda x: x.repeat(3, 1, 1)),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])
trans2=transforms.Compose([transforms.Resize(size=[32, 32]),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

])


svhn = torchvision.datasets.SVHN("./",transform=trans2,download=True)

mnistm = MNISTM("./",train=True,transform=trans2,download=True)

mnist=torchvision.datasets.MNIST("./",train=True,transform=trans,download=True)

syn_dig=Loadsyn()



model_path='generator/123_wo_syn.pth.tar'
gen = InterpolationGAN(args, device=device,is_train=False)
gen.G_ST.to(device)
gen.load(model_path)


fix_x=mnist[2][0].unsqueeze(0).cuda()




a=np.linspace(0.0,1.0,num=5)
b=np.linspace(0.0,1.0,num=5)
c=np.linspace(0.0,1.0,num=5)
current_code_list=[]

for a_item in a:
	for b_item in b:
		for c_item in c:
			if a_item+b_item+c_item==1:
				current_code_list.append([[a_item,b_item,c_item]])

for i,code in enumerate(current_code_list):
	current_code=torch.tensor(code)
	current_code=current_code.float().cuda()
	new_x=minibatch_transfer(gen,fix_x.cuda(),current_code)
	fp='generated_image/code_sample_'+str(code[0][0])+"_"+str(code[0][1])+"_"+str(code[0][2])+'.jpg'
	save_image((new_x+1.0)/2,fp)
