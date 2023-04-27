import matplotlib.pyplot as plt
import torch
import os
import torchvision.datasets
import torchvision.transforms as transforms
from torch import nn
from torchvision.models import vgg19


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.vgg = vgg19(pretrained=True)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.classifier[6] = nn.Sequential(nn.Linear(4096, 1), nn.Sigmoid())

    def forward(self, X):
        pred = self.vgg(X)
        return pred
