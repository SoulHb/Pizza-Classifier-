import matplotlib.pyplot as plt
import torch
import os
import torchvision.datasets
import torchvision.transforms as transforms
from torch import nn
from torchvision.models import vgg19


class VGG(nn.Module):
    def __init__(self):
        """
                Custom VGG model for binary classification.

                The VGG model is loaded with pre-trained weights, and the final fully-connected layer
                (classifier[6]) is replaced with a new layer for binary classification using sigmoid activation.

                """
        super(VGG, self).__init__()
        self.vgg = vgg19(pretrained=True)
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.classifier[6] = nn.Sequential(nn.Linear(4096, 1), nn.Sigmoid())

    def forward(self, X):
        """
                Forward pass of the VGG model.

                Args:
                    X (torch.Tensor): Input tensor.

                Returns:
                    torch.Tensor: Predicted probabilities for binary classification.

                """
        pred = self.vgg(X)
        return pred
