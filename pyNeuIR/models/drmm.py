import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.distance import CosineSimilarity
import math


def matching_histogram_mapping(query_tvs, doc_tvs, num_bins):
    # Local interaction

    bins = get_bins(num_bins)
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    histograms = [histogram_mapping([cos(query_tv,doc_tv) for doc_tv in doc_tvs],bins) for query_tv in query_tvs]
    return histograms

def get_bins(num_bins):
    bins = [-1.0 + (2*j)/(num_bins-1)  for j in range(num_bins-1)]
    bins.append(1.0)
    return bins

def histogram_mapping(values, bins):
    num_bins = len(bins)
    count = [0.0]*num_bins
    for value in values:
        bin_idx = int(((value.data[0]+1)*(num_bins-1)) / 2)
        count[bin_idx] += 1
    return count

def get_tv_idxs(word2ind, words, max_len=None):
    # deal with OOV
    # http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf
    if max_len:

        return Variable(torch.LongTensor([word2ind[w] if w in word2ind else word2ind['<unk>'] for w in words] +
        [word2ind['<pad>']] * (max_len - len(words))))
    else:
        return Variable(torch.LongTensor([word2ind[w] if w in word2ind else word2ind['<unk>'] for w in words]))


def term_gating_network(query_tvs, query_idxs, w):
    # apply exp
    exps = [(query_tv*w) for query_tv in query_tvs]
    norm = sum(exps)
    return [exp/norm for exp in exps]


def process_inputs(queries, docs_h, docs_l, word2ind, embeddings, num_bins=30):

    histograms_h = []
    histograms_l = []
    gates = []

    lens = [len(query) for query in queries]

    max_len = max(lens)

    for query, doc_h, doc_l in zip(queries,docs_h, docs_l):
        
        query_idxs = get_tv_idxs(word2ind, query, max_len)
        doc_h_idxs = get_tv_idxs(word2ind, doc_h)
        doc_l_idxs = get_tv_idxs(word2ind, doc_l)
        
        query_tvs = embeddings(query_idxs)
        doc_h_tvs = embeddings(doc_h_idxs)
        doc_l_tvs = embeddings(doc_l_idxs)

        histogram_h = matching_histogram_mapping(query_tvs,doc_h_tvs,num_bins)
        histogram_l = matching_histogram_mapping(query_tvs,doc_l_tvs,num_bins)

        gate = term_gating_network(query_tvs, query_idxs, w)
        histograms_h.append(histogram_h)
        histograms_l.append(histogram_l)
        gates.append(gate)


    histograms_h = Variable(torch.FloatTensor(histograms_h))
    histograms_l = Variable(torch.FloatTensor(histograms_l))
    gates = Variable(torch.FloatTensor(gates))
    return histograms_h, histograms_l,  gates

class FeedForwardNet(nn.Module):
    
    def __init__(self):
        super(FeedForwardNet, self).__init__()
        f1 = nn.Linear(30, 5)
        f2 = nn.Linear(5, 1)
        self.net = nn.Sequential (
            f1,
            nn.Tanh(),
            f2,
        )

    def forward(self, input):
        out = self.net(input)
        return 

class DRMM(nn.Module):

    def __init__(self, emb_dim):
        super(DRMM, self).__init__()
        # feedfoward net
        self.z = FeedForwardNet()
        # weights of gating network
        self.w = Variable(torch.FloatTensor(histograms_h))

    def forward(self, histograms, gates):  
        out_ffn = self.z(histograms)
        matching_score = out_ffn*gates
        return matching_score

class HingeLoss(torch.nn.Module):
    """
        Hinge Loss
    """
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, y_score, y_predict):
        return max(0, 1-y_score,y_predict)

def train(trainset):
    drmm = DRMM(0.005, [trainset.vocab_size,300,300,128])
    criterion = HingeLoss()
    optimizer = torch.optim.Adam(dssm.parameters(),lr = 0.0001)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True)

    model = DRMM()
    for i, data in enumerate(trainloader, 0):

        histograms_h, histograms_l, gates = process_inputs(queries, docs_h, docs_l, word2ind, embeddings, num_bins=30)
        out_h = drmm(histograms_h,gates)
        out_l = drmm(histograms_l,gates)
        optimizer.zero_grad()
        loss = criterion(out_h,out_l)
        loss.backward()
        optimizer.step()		
        if i %10 == 0 :
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss.data[0]))
    
    return drmm


def test(query,doc):
    net = DSSM().cuda()
    query = net(query)
    doc = net(doc)
    return query, doc