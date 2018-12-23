"""
Definition of Deep Models and Loss Functions
"""
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.append('../')


class ZhihuLiveDataset(Dataset):

    def __init__(self, X, y, transform=None):
        self.data = X
        self.labels = y
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'data': self.data.iloc[idx - 1].as_matrix().astype(np.float32),
                  'label': self.labels.iloc[idx - 1].as_matrix().astype(np.float32)}

        if self.transform:
            sample = self.transform(sample)

        return sample


class DoubanCommentsDataset(Dataset):
    """
    Douban Comments Dataset
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {'ft': self.X[idx], 'senti': self.y[idx]}

        return sample


class MTLoss(nn.Module):
    """
    Loss function for MTB-DNN
    """

    def __init__(self, k):
        super(MTLoss, self).__init__()
        self.k = k

    def forward(self, sum_loss):
        return torch.div(sum_loss, self.k)


class Branch(nn.Module):
    def __init__(self, params=None):
        """Constructs each branch necessary depending on input
        Args:
            b2(nn.Module()): An nn.Conv2d() is passed with specific params
        """
        super(Branch, self).__init__()
        if params is None:
            params = [8, 4, 1]
        self.bf1 = nn.Linear(params[0], params[1])
        self.bf2 = nn.Linear(params[1], params[2])

    def forward(self, x):
        return self.bf2(F.relu6(self.bf1(x)))


class MTNet(nn.Module):
    """
    definition of MTNet
    """

    def __init__(self):
        super(MTNet, self).__init__()
        self.layers = nn.Sequential(OrderedDict([
            ('fc1', nn.Sequential(nn.Linear(23, 16),
                                  nn.ReLU())),
            ('fc2', nn.Sequential(nn.Linear(16, 8),
                                  nn.ReLU())),
            ('fc3', nn.Sequential(nn.Linear(8, 8),
                                  nn.ReLU()))]))

        self.branch1 = Branch(params=[8, 4, 1])
        self.branch2 = Branch(params=[8, 3, 1])
        self.branch3 = Branch(params=[8, 5, 1])

    def forward(self, x):
        x1 = self.layers(x)
        x2 = torch.div(torch.add(torch.add(self.branch1(x1), self.branch2(x1)), self.branch3(x1)), 3)

        return x2
        # return (self.branch1(x1) + self.branch2(x1) + self.branch3(x1)) / 3
        # return torch.min([self.branch1(x1), self.branch2(x1), self.branch3(x1)])


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(23, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 8)
        # self.drop1 = nn.Dropout2d(0.7)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = self.drop1(x)
        x = self.fc4(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
