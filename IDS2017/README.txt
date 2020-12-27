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

__all__ = ('MPL19')

model_urls = {
('MLP19', 'IDS2017'): 'https://github.com/Kubernet2020/Explainability-for-model/blob/main/Bofan/feature19_classes13_lr001_batch10000.pth'
}

class MLP19(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 20)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(20, 20)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(20, 13)
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

def _MPL19(arch, *args, pretrained=False, progress=True, dataset='IDS2017', **kwargs):
    model = MLP(19)
    model = get_pretrained_model(arch, dataset, model, model_urls,
        pretrained=pretrained, progress=progress)
    return model

def MPL19(pretrained=False, progress=True, **kwargs):
    return _MPL19('MPL19', pretrained=pretrained, progress=progress, **kwargs)
