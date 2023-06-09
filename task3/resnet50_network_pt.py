import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class MyResnet50(nn.Module):
    """ modified resnet50 adapted for cifar 10 dataset
        INPUT: batches of images i.e. batch_size by 3 by image height by image width (together with target one hot labels of shape batch_size by 10 during training). Where 3 is for the 3 rgb channels
        and 10 becauys eof the 10 classes.
        OUTPUT: evaluation of network on input  of shape batch_size by 10
        data type either torch.float32 or torch.FloatTensor depending on cpu or gpu usage"""
    def __init__(self):
        super().__init__()
        self.modified50 = torchvision.models.resnet50(pretrained=True) #pretrained weigths
        self.modified50.fc = nn.Linear(2048,10) #original final fully connected layer was 2048 to 1000, modified to be 2048 to 10 for cifar 10

    def forward(self, x):
        x = self.modified50(x)
        return x