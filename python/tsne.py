# Numpy
import numpy as np

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

# Torchvision
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt

# OS
import os
import argparse
from collections import OrderedDict
from tqdm import tqdm
import json

# model
from model.resnet import *
from model.vgg import *
from model.LeNet import *

# loader
import loader

# plot
import plot

def get_config():
    # parser
    p = argparse.ArgumentParser(description="Train ResNet")
    p.add_argument('--gpu_id', type=int, default=0)

    p.add_argument('--model', type=str, default='res')
    p.add_argument('--batch_size', type=int, default=100)
    
    p.add_argument('--data', type=str, default='gen_mnist_test')    
    p.add_argument('--output_folder', type=str, default='./output')
    p.add_argument('--best_epoch', type=int, default=80)

    
    config = p.parse_args()
    
    return config

class Classifier():

    def __init__(self, config):
        self.config = config

    def test(self, model, test_loader):
        model.eval()
        
        valid_loss, len_data = 0, 0
        correct = 0
        loop = tqdm(test_loader)

        for i, (inputs, targets) in enumerate(loop):
            inputs = inputs.cuda(config.gpu_id)
            targets = targets.cuda(config.gpu_id)

            out, penultimate = model(inputs)
            if i == 0 :
                temp = penultimate.cpu().detach().numpy()
            else :
                temp = np.vstack((temp, penultimate.cpu().detach().numpy()))
            len_data += len(inputs)
            
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()
            # import ipdb; ipdb.set_trace()
            loop.set_description("Test")
            loop.set_postfix(
                OrderedDict(
                    {
                        "acc": f"{(correct / len_data):.4f}",
                    }
                )
            )
            loop.update()
        return temp

def main(config):
    # Create model
    if config.model == 'res':
        model = ResNet18()
    elif config.model == 'vgg':
        model = vgg16()
    elif config.model == 'lenet':
        model = LeNet5()
        
    model = model.cuda(config.gpu_id)
    cls = Classifier(config)

    # Load data
    test_loader = loader.test_DataSet(config)
    # import ipdb; ipdb.set_trace()
    model.load_state_dict(torch.load(os.path.join(config.output_folder, 'model_epoch_{:03d}.ckpt'.format(config.best_epoch))))
    
    penultimate = cls.test(model, test_loader)
    
    perplexity = [5, 10, 30, 40, 50]
    
    for per in perplexity:
        plot.tsne(penultimate, test_loader.dataset.targets, config, per)
    
    
    
if __name__ == '__main__':
    config = get_config()

    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)        

    main(config)  