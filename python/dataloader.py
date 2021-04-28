import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def train_loader(data_root, data, batch_size):

    if data == 'cifar':
        data_path = os.path.join(data_root, 'cifar-bench', 'train', 'cifar40')

        mean = [0.507, 0.487, 0.441]
        stdv = [0.267, 0.256, 0.276]

        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)
        ])

    elif data == 'imgnet':
        data_path = os.path.join(data_root, 'imgnet-bench', 'train', 'imagenet200')

        mean = [0.485, 0.456, 0.406]
        stdv = [0.229, 0.224, 0.225]
    
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)
        ])


    train_dataset = datasets.ImageFolder(data_path,
                                         transform=train_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)
    return train_loader


def test_loader(data_root, data, batch_size, mode, transform=False):

    if data == 'cifar':
        data_path = os.path.join(data_root, 'cifar-bench', f'{mode}', 'cifar40', 'labels')

        mean = [0.507, 0.487, 0.441]
        stdv = [0.267, 0.256, 0.276]

        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean,
                                                                  std=stdv)])
        if transform == True:
            test_transform = transforms.Compose(
                [transforms.RandomCrop((32, 32), padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=mean, std=stdv)])

    elif data == 'imgnet':
        data_path = os.path.join(data_root, 'imgnet-bench', f'{mode}', 'imagenet200', 'labels')

        mean = [0.485, 0.456, 0.406]
        stdv = [0.229, 0.224, 0.225]

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=stdv)])
        if transform == True:
            test_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=stdv)
            ])

    test_dataset = datasets.ImageFolder(data_path,
                                        transform=test_transform)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    test_targets = test_dataset.targets

    return test_loader, test_targets



def in_dist_loader(data_root, data, batch_size, mode, transform=False):

    if data == 'cifar':
        data_path = os.path.join(data_root, 'cifar-bench', f'{mode}', 'cifar40', 'labels')

        mean = [0.507, 0.487, 0.441]
        stdv = [0.267, 0.256, 0.276]

        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean,
                                                                  std=stdv)])
        if transform == True:
            test_transform = transforms.Compose(
                [transforms.RandomCrop((32, 32), padding=4),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=mean, std=stdv)])

    elif data == 'imgnet':
        data_path = os.path.join(data_root, 'imgnet-bench', f'{mode}', 'imagenet200', 'labels')

        mean = [0.485, 0.456, 0.406]
        stdv = [0.229, 0.224, 0.225]

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,
                                 std=stdv)])
        if transform == True:
            test_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=stdv)
            ])


    in_dataset = datasets.ImageFolder(data_path,
                                      transform=test_transform)

    # torch.manual_seed(1234)
    # np.random.seed(1234)
    # torch.cuda.manual_seed(1234)
    # torch.cuda.manual_seed_all(1234)
    in_loader = torch.utils.data.DataLoader(in_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=4)
    # np.random.seed(seed=None)

    return in_loader


def out_dist_loader(data_root, data, batch_size, mode, transform=False):



    if (data=='cifar60') or (data=='tiny-imagenet158') or (data=='svhn') or (data=='lsun-fix'):
    # if data_root.split('/')[-1].split('-')[0] == 'cifar':
        data_path = os.path.join(data_root, 'cifar-bench', mode, data, 'no-labels')
        mean = [0.507, 0.487, 0.441]
        stdv = [0.267, 0.256, 0.276]
        if data == 'new-tinyimagenet158':
            test_transform = transforms.Compose([transforms.Resize((32, 32)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=mean,
                                                                  std=stdv)])
            if transform == True:
                test_transform = transforms.Compose([transforms.Resize((32,32)),
                                                 transforms.RandomCrop((32,32), padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=mean,
                                                                      std=stdv)])
        else:
            test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean,
                                                                  std=stdv)])
            if transform == True:
                test_transform = transforms.Compose([transforms.RandomCrop((32,32), padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=mean,
                                                                      std=stdv)])
                                                 
                                                 
    if (data=='near-imagenet200') or (data=='external-imagenet394') or (data=='food101') or (data=='caltech256') or (data=='places365'):
    #elif data_root.split('/')[-1].split('-')[0] == 'imgnet':
        data_path = os.path.join(data_root, 'imgnet-bench', mode, data, 'no-labels')
        mean = [0.485, 0.456, 0.406]
        stdv = [0.229, 0.224, 0.225]

        test_transform = transforms.Compose([transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)])
        if transform == True:
            test_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            

    out_dataset = datasets.ImageFolder(data_path,
                                       transform=test_transform)

    # torch.manual_seed(1234)
    # np.random.seed(1234)
    # torch.cuda.manual_seed(1234)
    # torch.cuda.manual_seed_all(1234)
    out_loader = torch.utils.data.DataLoader(out_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=4)
    # np.random.seed(seed=None)

    return out_loader
