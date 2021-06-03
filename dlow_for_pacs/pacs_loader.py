import os 
import numpy as np 
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torch.utils import data

'''
class AddDomainLabel:
    def __init__(self,domain_label):
        self.domain_label=domain_label
    

    def __call__(self, label):

        return [label,self.domain_label]
'''


def spilit_dataset(dataset):

    dataset_size1=int(0.5*len(dataset))
    dataset_size2=len(dataset)-dataset_size1
    dataset1, dataset2 = torch.utils.data.random_split(dataset, [dataset_size1, dataset_size2])

    return dataset1,dataset2



root_dic={'p':'kfold/photo','a':'kfold/art_painting','c':'kfold/cartoon','s':'kfold/sketch'}



def get_train_dataset(list_train_domains,transform):
	dataset_list=[]

	for (domain_index,domain) in enumerate(list_train_domains):
		temp_dataset = datasets.ImageFolder(root = root_dic[domain],transform=transform
					)
		dataset_list.append(temp_dataset)
	#multiple_dataset=torch.utils.data.ConcatDataset(dataset_list)
	return dataset_list


def get_loader(list_train_domains,crop_size,image_size,batch_size,mode,num_workers):
	transform=[]
	if mode=='train':
		transform.append(transforms.RandomHorizontalFlip())
		transform.append(transforms.CenterCrop(crop_size))
	transform.append(transforms.Resize(image_size))
	transform.append(transforms.ToTensor())
	transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
	transform = transforms.Compose(transform)

	dataset_list=get_train_dataset(list_train_domains,transform)

	split_source,split_target=spilit_dataset(dataset_list[0])

	used_data_list=[split_source,split_target,dataset_list[1],dataset_list[2]]

	loader_list=[]

	for dataset in used_data_list:
		loader_list.append(data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=(mode=='train'),num_workers=num_workers))
	return loader_list 


def get_test_dataset(test_domain):

	temp_dataset = datasets.ImageFolder(root = root_dic[test_domain])
	return temp_dataset




if __name__ == '__main__':
	multiple_dataset =get_loader(['p','a','c','s'],256,256,32,'train',0)
	#train_set, val_set = torch.utils.data.random_split(multiple_dataset, [6000, 1299])
	from torchvision.utils import save_image
	print('123')
	i=0	
	for x,y in multiple_dataset[0]:
		#print(torch.max(x))
		#print(torch.min(x))
		x=x.cuda()
		y=y.cuda()
		print(y)
		i+=1
		if i>50:
			break







