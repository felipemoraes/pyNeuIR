import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.distance import CosineSimilarity
import numpy as np


class DSSM(nn.Module):

    def __init__(self):
        super(DSSM, self).__init__()
   
        self.net = nn.Sequential (
            nn.Linear(500*1, 30*10),
            nn.Tanh(),
            nn.Linear(30*10, 300),
            nn.Tanh(),
            nn.Linear(300, 128)
        )

        self.gamma = 0.005

    def forward(self, query, docs):
        query = self.forward_once(query)

        r_outputs = [self.cosine(query,self.forward_once(doc)) for doc in docs]
        
        soft_exps = [self.gamma*torch.exp(r_output) for r_output in r_outputs]

        norm = sum(soft_exps)
        output = torch.exp(soft_exps[0])/norm
        return output

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