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
		net = DSSM(0.005, [500,200,200,128])
		optimizer = torch.optim.SGD(net.parameters(),lr = 0.1, momentum=0.9)
		loss = LogLoss()
  
		query = Variable(torch.randn(100, 500))

		docs = [Variable(torch.randn(100, 500)) for _ in range(5)]

		output = net(query,docs[0],docs[1],docs[2],docs[3],docs[4])
	
		self.assertLessEqual(output[0].data.numpy()[0],1)
		self.assertGreaterEqual(output[0].data.numpy()[0],0)

		optimizer.zero_grad()
		loss_log = loss(output)
		print(loss_log)
		loss_log.backward()
		optimizer.step()

if __name__ == '__main__':
    unittest.main()

print(output)
