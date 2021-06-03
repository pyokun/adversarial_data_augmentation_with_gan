#a function, given data, code, return new data
import torch
from tensor_to_dataset import TensorToDataset


def domain_transfer(G,dataloader,code):
	temp_x=[]
	temp_y=[]
	with torch.no_grad():
		for index,(x,y) in enumerate(dataloader):
			if x.size()[1]==1:
				x=x.repeat(1,3,1,1)
			x=x.cuda()
			y=y.cuda()
			recon_x,_,_=G(x,code.expand(x.size()[0],code.size()[0]))
			recon_x=recon_x.detach().cpu()
			temp_x.append(recon_x)
			temp_y.append(y)

		new_x_set=torch.cat(temp_x)
		new_y_set=torch.cat(temp_y)
		result_dataset=TensorToDataset(new_x_set,new_y_set)
	return result_dataset


def minibatch_transfer(G,x,code):
	with torch.no_grad():
		recon_x=G(x,code)
	return recon_x








'''

if __name__ == '__main__':
	from torchvision import datasets,transforms
	from torch.utils.data import DataLoader,Dataset
	from torchvision.utils import save_image

	vae_path="vae"
	vae=torch.load(vae_path).cuda()
	vae.eval()
	trans = transforms.Compose([transforms.Resize(size=[32, 32]),transforms.ToTensor()])

	mnist_dataset = datasets.MNIST("/",train=True,transform=trans,download=True)
	tgt_loader=DataLoader(mnist_dataset,batch_size=16,shuffle=True, drop_last=True)	
	new_dataset=domain_transfer(vae,tgt_loader,code=torch.tensor([0.5,0.5]).cuda())
	for x,y in new_dataset:
		print(x.size())
		print(y)
		save_image(x.cpu(),'recon_mnist.png',nrow=1)
		break

'''