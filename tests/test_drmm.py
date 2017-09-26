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
        torch.manual_seed(222)
        drmm = DRMM(1,False)

        np.random.seed(222)
        queries_idfs = Variable(torch.FloatTensor(np.random.rand(20,2,1)))
        histograms_l = Variable(torch.FloatTensor(np.random.rand(20,2,30)))
        print(drmm(histograms_l,queries_idfs))

        #
        # idfs = load_idfs("/Users/felipemoraes/Developer/data/idfs.txt",5)
if __name__ == '__main__':
    unittest.main()

print(output)
