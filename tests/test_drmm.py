import os
import sys

sys.path.append('../pyNeuIR/')
import unittest

from pyNeuIR.models.drmm import DRMM, FeedForwardNet
from pyNeuIR.utils.data_preprocess import *
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


class DRMMTest(pyNeuIRTest):

    def test_output(self):
        queries = [["car", "rent"], ["rent"]]
        docs = [["car", "rent", "truck", "bump", "injunction",  "runway"],["rent"]]
        word2id, id2word = construct_vocab(docs,vocab_size=10)
        embeddings = load_word_embeddings(len(word2id), 5)

        input_histograms = get_histogram_minibatch(queries[0],docs[0], word2id,  embeddings, 5, False)
        
        print(input_histograms)
       
        #queries_tvs, queries_lens, queries_mask  = get_minibatch(queries,word2id, 10, False)
        
        #docs_tvs, docs_lens, docs_mask = get_minibatch(docs,word2id, 20, False)
        #print(queries_tvs)
        #model = DRMM(len(word2id), 5)
        #out = model(queries_tvs, docs_tvs)

if __name__ == '__main__':
    unittest.main()

print(output)
