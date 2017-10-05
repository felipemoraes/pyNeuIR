import os
import sys

sys.path.append('../pyNeuIR/')
import unittest

from pyNeuIR.models.drmm import DRMM, HingeLoss
from pyNeuIR.utils.preprocess import load_idfs
import torch
import torch.nn as nn
import gensim
from torch.nn import Parameter
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np

def get_model_size(model):
    return sum([ p.size(0) if len(p.size()) == 1 else p.size(0)*p.size(1) for p in model.parameters()])


class pyNeuIRTest(unittest.TestCase):

    def setUp(self):
        if hasattr(sys, 'gettotalrefcount'):
            gc.collect()
            self.ref_count_before_test = sys.gettotalrefcount()

    def tearDown(self):
        # For debugging purposes.
        if hasattr(sys, 'gettotalrefcount'):
            gc.collect()
            print(self._testMethodName,
                  sys.gettotalrefcount() - self.ref_count_before_test)


class DRMMTest(pyNeuIRTest):


    def test_output(self):  
        torch.manual_seed(222)
        #drmm = DRMM(1)

        np.random.seed(222)
        values = torch.FloatTensor(np.random.rand(2,3,1))
        values_sampled = torch.FloatTensor(values[:,:1,:])

        z1 = nn.Linear(1, 5)
        z2 = nn.Linear(5, 1)
        
        out_ffn = z1(Variable(values))
        out_ffn = F.tanh(out_ffn)
        out_ffn = z2(out_ffn)
        out_ffn = F.tanh(out_ffn).squeeze()
        #print(out_ffn)

        out_ffn = z1(Variable(values_sampled))
        out_ffn = F.tanh(out_ffn)
        out_ffn = z2(out_ffn)
        out_ffn = F.tanh(out_ffn).squeeze()
        #print(out_ffn)
        

        queries_idfs = Variable(torch.FloatTensor(np.random.rand(2,2,2)))
        #histograms_l = Variable(torch.FloatTensor(np.random.rand(2,5,30)))
        #print(get_model_size(drmm))
        #print(drmm(histograms_l,queries_idfs))
        f = nn.Linear(2,1,bias=False)
        #print(f.weight)
        #print(queries_idfs)
        print(f(queries_idfs).squeeze())
        print(F.softmax(f(queries_idfs).squeeze()))
        #
        # idfs = load_idfs("/Users/felipemoraes/Developer/data/idfs.txt",5)
if __name__ == '__main__':
    unittest.main()

print(output)
