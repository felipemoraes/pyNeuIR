import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.distance import CosineSimilarity
import numpy as np


class DSSM(nn.Module):

    def __init__(self, gamma, layers_len ):
        
        super(DSSM, self).__init__()
   
        self.net = nn.Sequential (
            nn.Linear(layers_len[0], layers_len[1]),
            nn.Tanh(),
            nn.Linear(layers_len[1], layers_len[2]),
            nn.Tanh(),
            nn.Linear(layers_len[2], layers_len[3]),
            nn.Tanh(),
            nn.Linear(layers_len[3], layers_len[4])
        )

        self.gamma = gamma

    def forward(self, query, pos_doc, neg_docs):
        
        query = self.forward_once(query)

        r_pos_output = self.cosine(query, self.forward_once(pos_doc))

        r_neg_outputs = [self.cosine(query,self.forward_once(neg_doc)) for neg_doc in neg_docs]

        soft_exp_pos = self.gamma*torch.exp(r_pos_output)

        soft_exps_neg = [self.gamma*torch.exp(r_neg_output) for r_neg_output in r_neg_outputs]


        norm_factor = soft_exp_pos + sum(soft_exps_neg)

        prob_query_given_pos_doc = soft_exp_pos/norm_factor
        
        return prob_query_given_pos_doc

    def forward_once(self, x):
        output = self.net(x)
        return output


    def cosine(self,x,y):
        cos = nn.CosineSimilarity()
        output = cos(x, y)
        return output


class LogLoss(torch.nn.Module):
    """
    Log loss function.
    Based on: http://dl.acm.org/citation.cfm?id=2505665
    """
    def __init__(self):
        super(LogLoss, self).__init__()

    def forward(self, output):
        return (-1)*torch.log(output)


def train(self, X):
    counter = []
    loss_history = [] 
    iteration_number= 0

    net = DSSM().cuda()
    criterion = LogLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005)
    for epoch in range(0,Config.train_number_epochs):
        for i, data in enumerate(X,0):
            query, docs = data
            query = Variable(query).cuda()
            docs = [Variable(doc).cuda() for doc in docs]
            output = net(query,docs)
            optimizer.zero_grad()
            loss_log = criterion(output)
            loss_log.backward()
            optimizer.step()

            if i %10 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.data[0]))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.data[0])
def test(query,doc):
    net = DSSM().cuda()
    query = net(query)
    doc = net(doc)
    return query, doc