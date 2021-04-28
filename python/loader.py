import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
import numpy as np

def in_dist_DataSet(config):

    if config.data == 'gen_mnist1' or config.data == 'gen_mnist2':
        trn_transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.RandomCrop(80, padding=0),
            transforms.ToTensor(),
            transforms.Normalize((0.1,), (0.2,)),
        ])

        tst_transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize(80),
            transforms.ToTensor(),
            transforms.Normalize((0.1,), (0.2,)),
        ])
    elif config.data == 'ori_mnist1' or config.data == 'ori_mnist2':
        trn_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
                                             
        tst_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
    
    if config.data == 'gen_mnist1':
        trn_dataset = datasets.ImageFolder('/daintlab/home/cch/Data/MNIST_V2/mnist-generated-exp1/cls-trn-data', transform=trn_transform)
        val_dataset = datasets.ImageFolder('/daintlab/home/cch/Data/MNIST_V2/mnist-generated-exp1/cls-tst-data', transform=tst_transform)
    
    elif config.data == 'gen_mnist2':
        trn_dataset = datasets.ImageFolder('/daintlab/home/cch/Data/MNIST_V2/mnist-generated-exp2/cls-trn-data', transform=trn_transform)
        val_dataset = datasets.ImageFolder('/daintlab/home/cch/Data/MNIST_V2/mnist-generated-exp2/cls-tst-data', transform=tst_transform)
        
    elif config.data == 'ori_mnist1':
        trn_dataset = datasets.ImageFolder('/daintlab/home/jihyokim/projects/ood/results/mnist-exp1/data/cls-trn-data', transform=trn_transform)
        val_dataset = datasets.ImageFolder('/daintlab/home/jihyokim/projects/ood/results/mnist-exp1/data/cls-tst-data', transform=tst_transform)
    
    elif config.data == 'ori_mnist2':
        trn_dataset = datasets.ImageFolder('/daintlab/home/jihyokim/projects/ood/results/mnist-exp2/data/cls-trn-data', transform=trn_transform)
        val_dataset = datasets.ImageFolder('/daintlab/home/jihyokim/projects/ood/results/mnist-exp2/data/cls-tst-data', transform=tst_transform)
        
    trn_loader = torch.utils.data.DataLoader(trn_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=True,
                                             num_workers=4)

    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=4)

    return trn_loader, val_loader

def test_DataSet(config):

    if config.data == 'gen_mnist_test':
        trn_transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.RandomCrop(80, padding=0),
            transforms.ToTensor(),
            transforms.Normalize((0.1,), (0.2,)),
        ])

        tst_transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize(80),
            transforms.ToTensor(),
            transforms.Normalize((0.1,), (0.2,)),
        ])
    elif config.data == 'ori_mnist_test':
        trn_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
                                             
        tst_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
    
    if config.data == 'gen_mnist_test':
        tst_dataset = datasets.ImageFolder('/daintlab/home/cch/Data/mnist-generated/test', transform=tst_transform)
        # tst_dataset = datasets.ImageFolder('/daintlab/home/cch/Data/MNIST_V2/mnist-generated-exp1/ood-tst-data', transform=tst_transform)

    elif config.data == 'ori_mnist_test':
        tst_dataset = datasets.MNIST("/daintlab/home/cch/Data/", 
                                     download=False,
                                     train=False,
                                     transform= tst_transform)
                                     

    
    tst_loader = torch.utils.data.DataLoader(tst_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=4)

    return tst_loader


def even_number(dataset):
    dataset.data = torch.cat([dataset.data[dataset.targets==0],
                                dataset.data[dataset.targets==2],
                                dataset.data[dataset.targets==4],
                                dataset.data[dataset.targets==6],
                                dataset.data[dataset.targets==8]])
    
    dataset.targets = torch.cat([dataset.targets[dataset.targets==0],
                                    dataset.targets[dataset.targets==2],
                                    dataset.targets[dataset.targets==4],
                                    dataset.targets[dataset.targets==6],
                                    dataset.targets[dataset.targets==8]])
    return dataset
    
def odds_number(dataset):
    dataset.data = torch.cat([dataset.data[dataset.targets==1],
                                dataset.data[dataset.targets==3],
                                dataset.data[dataset.targets==5],
                                dataset.data[dataset.targets==7],
                                dataset.data[dataset.targets==9]])
    
    dataset.targets = torch.cat([dataset.targets[dataset.targets==1],
                                    dataset.targets[dataset.targets==3],
                                    dataset.targets[dataset.targets==5],
                                    dataset.targets[dataset.targets==7],
                                    dataset.targets[dataset.targets==9]])
    return dataset