import torch
from torchvision import datasets
from torchvision import transforms
from mnist_m import MNISTM
from syn_digits import Loadsyn
from torch.utils import data


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

    svhn_loader = data.DataLoader(dataset=svhn,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)

    mnist_loader = data.DataLoader(dataset=mnist,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)
    mnist_m_loader=data.DataLoader(dataset=mnistm,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)

    syn_loader=data.DataLoader(dataset=syn_dig,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)



    return [mnist_loader, svhn_loader,mnist_m_loader,syn_loader]




