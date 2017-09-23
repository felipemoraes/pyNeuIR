import os
import sys

sys.path.append('../pyNeuIR/')
import unittest

from pyNeuIR.models.drmm import DRMM_TV, DRMM_IDF, HingeLoss
from pyNeuIR.utils.pairs_generator import PairsGenerator
from pyNeuIR.utils.preprocess import process_minibatch
import torch
import torch.nn as nn
import gensim
from torch.nn import Parameter


from torch.autograd import Variable
import numpy as np



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
        drmm = DRMM_TV(use_gpu=False)
        queries_tvs = Variable(torch.FloatTensor(np.random.rand(20,5,300)))
        histograms_l = Variable(torch.FloatTensor(np.random.rand(20,5,30)))
        print(drmm(histograms_l,queries_tvs))

if __name__ == '__main__':
    unittest.main()

print(output)
