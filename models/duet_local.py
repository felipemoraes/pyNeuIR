import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F

class DuetLocal(nn.Module):

    def __init__(self, words_per_query, words_per_doc):
        super(DuetLocal, self).__init__()
        
        num_hidden_nodes = 300

        self.local_conv1 = nn.Conv2d(1, num_hidden_nodes, (1,words_per_doc),stride=(1,1))   

        self.local_linear1 = nn.Linear(num_hidden_nodes*words_per_query, num_hidden_nodes)     
        self.local_linear2 = nn.Linear(num_hidden_nodes, num_hidden_nodes)
        self.local_dropout = nn.Dropout(p=0.2)  
        self.local_linear3 = nn.Linear(num_hidden_nodes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.xavier_uniform(m.weight.data, gain=1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                weight_init.xavier_uniform(m.weight.data, gain=1)
                m.bias.data.zero_()
        
    def forward(self, features_local, num_docs):  
        scores = [self.forward_once(features_local[i]) for i in range(num_docs)]
        scores = torch.stack(scores).squeeze()
        if len(scores.size()) > 1:
            return scores.permute(1,0)
        return scores
    
    def forward_once(self, features_local):
        
        out_local = self.local_conv1(features_local).squeeze()
        out_local = F.tanh(out_local)
        
        out_local = self.local_linear1(out_local.view(out_local.size(0), -1))
        out_local = F.tanh(out_local)
        
        out_local = self.local_linear2(out_local)
        out_local = F.tanh(out_local)
        
        out_local = self.local_dropout(out_local)
        
        out_local = self.local_linear3(out_local)
        score = F.tanh(out_local)

        return score.squeeze()