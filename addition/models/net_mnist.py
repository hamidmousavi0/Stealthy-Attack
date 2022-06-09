import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantization import *


import numpy as np
import torch
import torch.nn as nn
from .quantization import *
__all__=['Net','small_cnn']
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = quan_Conv2d(1, 10, kernel_size=5)
        self.conv2 = quan_Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = quan_Linear(320, 50)
        self.fc2 = quan_Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

def small_cnn(pretrained=False, **kwargs):

    model = Net(**kwargs)
    if pretrained:
        model = torch.load("/media/hamid/ali/RESEARCH/DeepFool_weight_attack(ICCAD)/model_small_robust.pth")

    return model