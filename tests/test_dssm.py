import os
import sys

sys.path.append('../pyNeuIR/')
import unittest

from pyNeuIR.models.dssm import DSSM, LogLoss
import torch

from torch.autograd import Variable
import torch
from torch.autograd import Variable




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

class DSSMTest(pyNeuIRTest):

	def test_output(self):
		net = DSSM(0.005, [500,200,200,200,128])

		criterion = LogLoss()
  
		query = Variable(torch.randn(1, 500*1))

		pos_doc = Variable(torch.randn(1, 500*1))

		neg_docs = [Variable(torch.randn(1, 500*1)) for _ in range(4)]

		output = net(query,pos_doc,neg_docs)

		self.assertLessEqual(output.data.numpy()[0],1)
		self.assertGreaterEqual(output.data.numpy()[0],0)


if __name__ == '__main__':
    unittest.main()

print(output)
