import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.distance import CosineSimilarity
import math

class DSSM(nn.Module):

    def __init__(self, gamma, layers_len ):
        
        super(DSSM, self).__init__()
        x = nn.Linear(layers_len[0], layers_len[1])
        f1 = nn.Linear(layers_len[1], layers_len[2])
        f2 = nn.Linear(layers_len[2], layers_len[3])
        
        """
            Initiliaze network weights a = \sqrt(6/(fanin+fanout)), 
            where fanin = input length and fanout output length
            Based on: http://dl.acm.org/citation.cfm?id=2505665
        """

        a = init_value(layers_len[0],layers_len[1])
        nn.init.uniform(x.weight,(-1)*a, a)
        a = init_value(layers_len[1],layers_len[2])
        nn.init.uniform(f1.weight,(-1)*a, a)
        a = init_value(layers_len[2],layers_len[3])
        nn.init.uniform(f2.weight,(-1)*a, a)

        self.net = nn.Sequential (
            x,
            nn.Tanh(),
            f1,
            nn.Tanh(),
            f2
        )

        self.gamma = gamma

    def forward(self, query, doc1, doc2, doc3, doc4, doc5):
        
        query = self.forward_once(query)

        docs = [self.forward_once(doc) for doc in [doc1, doc2, doc3, doc4, doc5]]

        r_outputs = [self.cosine(query,doc) for doc in docs]

        soft_exps = [self.gamma*torch.exp(r_output) for r_output in r_outputs]

        norm_factor = sum(soft_exps)

        prob_query_given_pos_doc = soft_exps[0]/norm_factor
        
        return prob_query_given_pos_doc

    def forward_once(self, x):
        output = self.net(x)
        return output


    def cosine(self,x,y):
        cos = nn.CosineSimilarity()
        output = cos(x, y)
        return output

def init_value(fanin, fanout):
    return math.sqrt(6/(fanin+fanout))

class LogLoss(torch.nn.Module):
    """
    Log loss function.
    Based on: http://dl.acm.org/citation.cfm?id=2505665
    """
    def __init__(self):
        super(LogLoss, self).__init__()

    def forward(self, output):
        loss = -1* torch.log(output)
        loss = torch.mean(loss) 
        return loss


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