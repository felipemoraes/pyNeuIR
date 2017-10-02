import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.nn.init as weight_init
torch.manual_seed(222)

use_cuda = torch.cuda.device_count() > 0

if use_cuda:
    torch.cuda.manual_seed(222)

class DRMM(nn.Module):

    def __init__(self, dim_term_gating, use_cuda=True):
        super(DRMM, self).__init__()
        
        # feedfoward matching network
        self.z1 = nn.Linear(30, 5)
        self.z2 = nn.Linear(5, 1)
        weight_init.xavier_normal(self.z1.weight, gain=weight_init.calculate_gain('tanh'))
        weight_init.xavier_normal(self.z2.weight, gain=weight_init.calculate_gain('tanh'))
        # term gating network

        self.g = nn.Linear(dim_term_gating,1)
        weight_init.xavier_normal(self.g.weight, gain=weight_init.calculate_gain('linear'))

        
    def forward(self, histograms, queries_tvs):  

        out_ffn = self.z1(histograms)
        out_ffn = F.tanh(out_ffn)
        out_ffn = self.z2(out_ffn).squeeze()

        out_tgn = F.softmax(self.g(queries_tvs).squeeze())

        matching_score = torch.sum(out_ffn * out_tgn,dim=1)
        return matching_score


class HingeLoss(torch.nn.Module):
    """
        Hinge Loss
          max(0, 1-x+y)
    """
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, x, y):
        output = 1-x+y
        return output.clamp(min=0).mean()