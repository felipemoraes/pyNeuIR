import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class TermGatingNet(nn.Module):
    
    def __init__(self, use_gpu=True, dim=300):
        super(TermGatingNet, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor(np.random.rand(dim)))
    
        if use_gpu:
            self.w = self.w.cuda()

    def forward(self, queries_tvs):
        softmax = nn.Softmax()
        w_view = self.w.unsqueeze(0).expand(queries_tvs.size(0), len(self.w)).unsqueeze(2)
        out = softmax(queries_tvs.bmm(w_view).squeeze(2))
        return out

class FeedForwardMatchingNet(nn.Module):
    
    def __init__(self):
        super(FeedForwardMatchingNet, self).__init__()
        f1 = nn.Linear(30, 5)
        f2 = nn.Linear(5, 1)
        self.net = nn.Sequential (
            f1,
            nn.Tanh(),
            f2,
        )

    def forward(self, input):
        out = self.net(input)
        return out

class DRMM_TV(nn.Module):

    def __init__(self, use_gpu=True):
        super(DRMM_TV, self).__init__()
        # feedfoward matching network
        self.z = FeedForwardMatchingNet()
        # term gating network
        self.g = TermGatingNet(use_gpu)
        
        
    def forward(self, histograms, queries_tvs):  
        out_ffn = self.z(histograms)
        out_tgn = self.g(queries_tvs)

        matching_score = torch.sum(out_ffn * out_tgn,dim=1)
        
        return matching_score

class DRMM_IDF(nn.Module):

    def __init__(self):
        super(DRMM_IDF, self).__init__()
        # feedfoward matching network
        self.z = FeedForwardMatchingNet()
        
    def forward(self, histograms, queries_idfs):  

        out_ffn = self.z(histograms)

        softmax = nn.Softmax()
        out_tgn = softmax(queries_idfs)

        matching_score = torch.sum(out_ffn * out_tgn,dim=1)
        
        return matching_score


class HingeLoss(torch.nn.Module):
    """
        Hinge Loss
    """
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, x, y):
        output = x.mul_(-1)
        output.add_(y)
        output.add_(1)
        output = output.clamp(min=0)
        output = output.sum()
        output = output / y.size(0)
        return output