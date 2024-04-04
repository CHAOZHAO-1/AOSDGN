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

