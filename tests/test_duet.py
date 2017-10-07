import os
import sys

sys.path.append('../pyNeuIR/')
import unittest


from pyNeuIR.models.duet import Duet
from pyNeuIR.utils.input_utils import *
import torch
import torch.nn as nn

from torch.nn import Parameter
import torch.nn.functional as F
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


class DuetTest(pyNeuIRTest):


    def test_output(self):  
        
        # words_per_query, words_per_doc, num_ngraphs
        duet = Duet(10, 1000, 2000)
        
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(duet.parameters(),lr = 0.01)

        target = Variable((torch.ones(8)-1).type(torch.LongTensor))
        print(target)
    
        queries_features_distrib = torch.stack([torch.rand((2000,10)) for _ in range(8)]).unsqueeze(1)
        queries_features_distrib = Variable(queries_features_distrib)

        all_scores = []
                
        for docs_pair in range(5):
            queryies_docs_features_local = torch.stack([torch.rand((1000,10)) for _ in range(8)]).unsqueeze(1)
            docs_features_distrib = torch.stack(torch.stack([torch.rand((2000,1000)) for _ in range(8)])).unsqueeze(1)

            queryies_docs_features_local = Variable(queryies_docs_features_local)
            docs_features_distrib = Variable(docs_features_distrib)
            scores = duet(queryies_docs_features_local, queries_features_distrib, docs_features_distrib)
            all_scores.append(scores)

        
        optimizer.zero_grad()
        all_scores = torch.stack(all_scores).permute(1,0)

        print(all_scores)
        loss = criterion(all_scores,target)
        print(loss)
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    unittest.main()

print(output)
