import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.distance import CosineSimilarity


class DSSM(nn.Module):

    def __init__(self):
        super(DSSM, self).__init__()
   
        self.net = nn.Sequential (
            nn.Linear(500*100, 30*100),
            nn.Tanh(inplace=True),
            nn.Linear(30*100, 300),
            nn.Tanh(inplace=True),
            nn.Linear(300, 128)
        )

    def forward(self, query, docs):
        query = self.forward_once(query)

        for i, doc in enumerate(docs):
            doc = self.forward_once(doc)
            doc = self.gamma*self.cosine(query,doc)

        docs = torch.exp(docs)
        norm = torch.sum(docs)
  
        output = torch.exp(docs.narrow(0,1))/norm
        return docs

    def forward_once(self, x):
        output = self.net(x)
        return output


    def cosine(self,x,y):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        output = cos(input1, input2)
        return output

class LogLoss(torch.nn.Module):
    """
    Log loss function.
    Based on: http://dl.acm.org/citation.cfm?id=2505665
    """

    def __init__(LogLoss):
        super(LogLoss, self).__init__()
       

    def forward(self, output):
        return (-1)*torch.log(output)
