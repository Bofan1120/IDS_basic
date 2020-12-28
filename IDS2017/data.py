import torch
import pandas as pd
import tensorflow as tf
import numpy as np
import torch.nn as nn
import glob
import time
from os.path import join
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
from torch.nn import functional as F
from torch.utils import data
import torchvision.datasets as datasets
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
from nbdt.utils import DATASET_TO_NUM_CLASSES, DATASETS
from collections import defaultdict
from nbdt.graph import get_wnids, read_graph, get_leaves, get_non_leaves, \
    FakeSynset, get_leaf_to_path, wnid_to_synset, wnid_to_name
from . import imagenet
import torch.nn as nn
import random

IDS2017URL = 

__all__ = names = ('IDS2017')

class IDS2017(Dataset):
    def __init__(self, IDS2017URL, tranform=True):
      df = pd.read_csv(IDS2017URL)
      df.loc[df['Label'].isin([ 'Web Attack - Sql Injection','Web Attack - XSS', 'Web Attack - Brute Force']), 'Label'] = 'Web Attack'
      df = df[['Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Flow IAT Max', 'Bwd Packet Length Std', 'Fwd Packet Length Max', 'Flow Bytes/s', 'Total Length of Bwd Packets', 'Fwd Packet Length Mean', 'Flow Duration', 'Flow IAT Min', 'Total Length of Fwd Packets', 'Flow IAT Mean', 'Total Backward Packets', 'Bwd Packet Length Max', 'Flow Packets/s', 'Flow IAT Std', 'Fwd IAT Total', 'Bwd Packet Length Min', 'Fwd Packet Length Min', 'Label']]
      
      #df = df[~df['Label'].isin(['Heartbleed', 'Web Attack - Sql Injection', 'Infiltration', 'Web Attack - XSS', 'Web Attack - Brute Force'])]
      df = df.replace([-np.inf, np.inf], np.nan)
      df = df.dropna()
      #df.drop(df.columns[[37,39,61,62,63,64,65,66]],axis=1,inplace=True)
      self.X = df.iloc[:, :-1].apply(lambda x: (x-np.mean(x))/np.std(x)).values
      self.y = df.values[:, -1]
      self.X = self.X.astype('float32')
      le = LabelEncoder()
      le.fit(["BENIGN", "DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye","FTP-Patator","SSH-Patator","DoS slowloris","DoS Slowhttptest","Web Attack","Bot","Infiltration","Heartbleed"])
      self.y = le.transform(self.y)


    def __len__(self):
      return len(self.X)
    
    def __getitem__(self, idx):
      return [self.X[idx], self.y[idx]]
 
    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])
