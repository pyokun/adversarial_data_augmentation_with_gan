import torch
from torchvision import datasets
from torchvision import transforms
from mnist_m import MNISTM
from syn_digits import Loadsyn
from torch.utils import data



def spilit_dataset(dataset):

    dataset_size1=int(0.5*len(dataset))
    dataset_size2=len(dataset)-dataset_size1
    dataset1, dataset2 = torch.utils.data.random_split(dataset, [dataset_size1, dataset_size2])

    return dataset1,dataset2







def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    
    transform1 = transforms.Compose([
                    transforms.Scale(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform2 = transforms.Compose([
                    transforms.Scale(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.5)),
                    transforms.Lambda(lambda x: x.repeat(3, 1, 1))

                    ])
    
    mnistm=MNISTM("./",train=True,transform=transform1,download=True)
    syn_dig=Loadsyn()
    svhn = datasets.SVHN(root=config.svhn_path, download=True, transform=transform1)
    mnist = datasets.MNIST(root=config.mnist_path, download=True, transform=transform2)

    dataset_dic={1:mnist,2:svhn,3:mnistm,4:syn_dig}

    split_source,split_target=spilit_dataset(dataset_dic[config.source_idx])


    output=[]

    for dataset in [split_source,split_target,dataset_dic[config.t1_idx],dataset_dic[config.t2_idx]]:
      output.append(data.DataLoader(dataset=dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers))


    return output


