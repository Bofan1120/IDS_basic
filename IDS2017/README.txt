import torch
import pandas as pd
import tensorflow as tf
import numpy as np
import torch.nn as nn
from numpy import vstack
from numpy import argmax
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from nbdt.models.utils import get_pretrained_model

__all__ = ('MPL75')

class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 76)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(76, 76)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(76, 10)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Softmax(dim=1)

    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)

        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)

        # output layer
        X = self.hidden3(X)
        X = self.act3(X)

        return X
def _MPL75(arch, *args, pretrained=False, progress=True, dataset='CIFAR10', **kwargs):
    model = MLP(75)
    model = get_pretrained_model(arch, dataset, model, model_urls='https://github.com/Bofan1120/IDS_basic/blob/master/IDS2017/2.pth',
        pretrained=pretrained, progress=progress)
    return model

def MPL75(pretrained=False, progress=True, **kwargs):
    return _MPL75('MPL75', pretrained=pretrained, progress=progress, **kwargs)
