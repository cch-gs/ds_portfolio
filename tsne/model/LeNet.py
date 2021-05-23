import torch.nn as nn
import torch

class LeNet5(torch.nn.Module):          
     
    def __init__(self):     
        super(LeNet5, self).__init__()
        # Convolution (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        # Max-pooling
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        # Convolution
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        # Max-pooling
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2) 
        # Fully connected layer
        self.fc1 = torch.nn.Linear(400, 120) 
        self.fc2 = torch.nn.Linear(120, 84)   
        self.fc3 = torch.nn.Linear(84, 5)     
        
        # self.fc1 = torch.nn.Linear(5184, 2592) 
        # self.fc2 = torch.nn.Linear(2592, 1296)   
        # self.fc3 = torch.nn.Linear(1296, 5)     

        
    def forward(self, x):
        # convolve, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.conv1(x))  
        # max-pooling with 2x2 grid 
        x = self.max_pool_1(x) 
        # convolve, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.conv2(x))
        # max-pooling with 2x2 grid
        x = self.max_pool_2(x)
        # first flatten 'max_pool_2_out' to contain 16*5*5 columns
        # read through https://stackoverflow.com/a/42482819/7551231
        x = x.view(x.size(0),-1)
        # FC-1, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.fc1(x))
        # FC-2, then perform ReLU non-linearity
        penultimate = torch.nn.functional.relu(self.fc2(x))
        # FC-3
        x = self.fc3(penultimate)
        
        return x, penultimate