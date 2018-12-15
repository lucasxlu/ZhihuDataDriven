"""
Definition of Deep Models and Loss Functions
"""
from collections import OrderedDict
import sys

import torch.nn as nn
import torch
import torch.nn.functional as F

sys.path.append('../')
from util.cfg import cfg


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
    def __init__(self, params=[8, 4, 1]):
        """Constructs each branch necessary depending on input
        Args:
            b2(nn.Module()): An nn.Conv2d() is passed with specific params
        """
        super(Branch, self).__init__()
        self.bf1 = nn.Linear(params[0], params[1])
        self.bf2 = nn.Linear(params[1], params[2])

    def forward(self, x):
        return self.bf2(F.tanh(self.bf1(x)))


class MTBDNN(nn.Module):
    def __init__(self, K=2):
        super(MTBDNN, self).__init__()
        self.K = K
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
        out = torch.zeros([cfg['batch_size'], 1])

        if torch.cuda.is_available():
            out = out.cuda()

        for idx, module in self.layers.named_children():
            x = F.tanh(module(x))

        temp = x

        return torch.div(torch.add(torch.add(self.branch1(temp), self.branch2(temp)), self.branch3(temp)), self.K)
        # return torch.min([self.branch1(temp), self.branch2(temp), self.branch3(temp)])


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
