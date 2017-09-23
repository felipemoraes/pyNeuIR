import operator
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from torch.autograd import Variable
import torch 

def padding(values, maxlen):
    if len(values) < maxlen:
        return values + ["<pad>"]* (maxlen-len(values)) 
    return values

def process_minibatch(histograms_h, histograms_l, queries, use_gpu=True):

   
    lens = [len(query) for query in queries]
    max_len = max(lens)
    queries_tvs = []
    for i, query in enumerate(queries):
        queries_tvs.append([])
        queries_tvs[i].extend(queries[i])
        queries_tvs[i].extend([np.zeros(300)] *( max_len - len(query)))

    lens = [len(histograms) for histograms in histograms_h]
    max_len = max(lens)
    
    histograms_h_c = []
    for i, histograms in enumerate(histograms_h):
        histograms_h_c.append([])
        histograms_h_c[i].extend(histograms)        
        histograms_h_c[i].extend([np.zeros(30)] *( max_len - len(histograms)))
    
    lens = [len(histograms) for histograms in histograms_l]
    max_len = max(lens)
    
    histograms_l_c = []
    for i, histograms in enumerate(histograms_l):
        histograms_l_c.append([])
        histograms_l_c[i].extend(histograms) 
        histograms_l_c[i].extend([np.zeros(30)] *( max_len - len(histograms)))

    histograms_h_c = Variable(torch.FloatTensor(histograms_h_c))
    histograms_l_c = Variable(torch.FloatTensor(histograms_l_c))
    queries_tvs = Variable(torch.FloatTensor(np.array(queries_tvs)))
    
    if use_gpu:
        histograms_h_c = histograms_h_c.cuda()
        histograms_l_c = histograms_l_c.cuda()
        queries_tvs = queries_tvs.cuda()
    return histograms_h_c, histograms_l_c, queries_tvs


def load_histograms(histogram_file):
    histograms = {}
    for line in open(histogram_file):
        qid, docno, hs = line.split(" ", 2)
        hs = [ np.array([float(v) for v in histogram.split()]) for histogram in hs.split("\t")]
        if not qid in histograms:
            histograms[qid] = {}
        histograms[qid][docno] = hs
    return histograms


class PreTrainedWordEmbeddings():

    def __init__(self, w2v_path, dim=300):
        self.w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        self.dim = dim
        self.oov = {}
        
    def __call__(self, word):
        if word == "<pad>":
            return np.zeros(self.dim)
        if word in self.w2v_model.wv.vocab:
            return self.w2v_model.wv[word]
        elif word in self.oov:
            return self.oov[word]
        else:
            # It deals with OOV like described here:
            # http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf
            self.oov[word] = np.random.rand(self.dim)
            return self.oov[word]
