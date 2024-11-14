import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torch
import itertools
import mmd
import torch.nn as nn
from utils import *
from torch.autograd import Variable
import  numpy as np
import torch.nn.functional as F
import Coral



def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                                  - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)

class CNN_1D(nn.Module):

    def __init__(self, num_classes=31):
        super(CNN_1D, self).__init__()
        # self.sharedNet = resnet18(False)
        # self.cls_fc = nn.Linear(512, num_classes)

        self.sharedNet = CNN()
        self.cls_fc = nn.Linear(256, num_classes)


    def forward(self, source):

        # source= source.unsqueeze(1)

        feature = self.sharedNet(source)
        pre=self.cls_fc(feature)

        return pre, feature


class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, num_classes=10):
        super(CNN, self).__init__()


        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64,stride=1),  # 32, 24, 24
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            )  # 32, 12,12     (24-2) /2 +1


        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=16,stride=1),  # 128,8,8
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))# 128, 4,4

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5,stride=1),  # 32, 24, 24
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )  # 32, 12,12     (24-2) /2 +1

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5,stride=1),  # 128,8,8
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4)
        )



        # self.fc = nn.Linear(256, num_classes)

    def forward(self, x):

        x = x.unsqueeze(1)
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)

        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)


        x = x.view(x.size(0), -1)

        # x = self.layer5(x)

        # x = self.fc(x)

        return x
