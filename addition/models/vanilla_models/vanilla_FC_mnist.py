import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import  datetime
__all__=['FC','FC_5']
class FC(nn.Module):

    def __init__(self, n_classes):
        super(FC, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(in_features=1024,out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=150),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=150, out_features=n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
def FC_5(pretrained=False, **kwargs):
    model = FC_5(**kwargs)
    if pretrained:
        model = torch.load("")

    return model

