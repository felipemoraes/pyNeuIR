import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.nn.init as weight_init
torch.manual_seed(222)

use_cuda = torch.cuda.device_count() > 0

if use_cuda:
    torch.cuda.manual_seed(222)

class Duet(nn.Module):

    def __init__(self, words_per_query, words_per_doc, num_ngraphs):
        super(Duet, self).__init__()
        
        num_hidden_nodes = 300
        word_window_size = 3
        pooling_kernel_width_query = words_per_query - word_window_size + 1 # = 8
        pooling_kernel_width_doc = 100

        num_pooling_windows_doc = ((words_per_doc - word_window_size + 1) - pooling_kernel_width_doc) + 1 # = 899        

        self.local_conv1 = nn.Conv2d(1, num_hidden_nodes, (words_per_doc,1),stride=(1,1))   

        self.local_linear1 = nn.Linear(num_hidden_nodes*words_per_query, num_hidden_nodes)     
        self.local_linear2 = nn.Linear(num_hidden_nodes, num_hidden_nodes)
        self.local_dropout = nn.Dropout(p=0.2)  
        self.local_linear3 = nn.Linear(num_hidden_nodes, 1)

        self.embed_q_conv =  nn.Conv2d(1,num_hidden_nodes, (num_ngraphs,word_window_size), stride=(1,1))
        self.embed_q_max_pool  = nn.MaxPool2d((1, pooling_kernel_width_query), stride=(1, 1))
        self.embed_q_linear = nn.Linear(num_hidden_nodes, num_hidden_nodes)  

        self.embed_d_conv1 =  nn.Conv2d(1,num_hidden_nodes, (num_ngraphs,word_window_size), stride=(1,1))
        self.embed_d_max_pool  = nn.MaxPool2d((1, pooling_kernel_width_doc), stride=(1, 1))
        self.embed_d_conv2 =  nn.Conv2d(1,num_hidden_nodes, (num_hidden_nodes,1), stride=(1,1))

        self.distrib_linear1 = nn.Linear(num_hidden_nodes*num_pooling_windows_doc, num_hidden_nodes)
        self.distrib_linear2 = nn.Linear(num_hidden_nodes, num_hidden_nodes)
        self.distrib_dropout = nn.Dropout(p=0.2)  
        self.distrib_linear3 = nn.Linear(num_hidden_nodes, 1)
        
        
    def forward(self, features_local, features_distrib_query, features_distrib_doc):  
        
        out_local = self.local_conv1(features_local).squeeze()
        out_local = F.tanh(out_local)
        
        out_local = self.local_linear1(out_local.view(out_local.size(0), -1))
        out_local = self.local_dropout(out_local)
        out_local = self.local_linear3(out_local)

        out_embed_q = self.embed_q_conv(features_distrib_query)
        out_embed_q = F.tanh(out_embed_q)
        out_embed_q = self.embed_q_max_pool(out_embed_q).squeeze()
        out_embed_q = F.tanh(out_embed_q)
        out_embed_q = self.embed_q_linear(out_embed_q)
        out_embed_q = F.tanh(out_embed_q)
        out_embed_q = out_embed_q.permute(1,0)


        out_embed_d = self.embed_q_conv(features_distrib_doc)
        out_embed_d = F.tanh(out_embed_d)
        out_embed_d = self.embed_d_max_pool(out_embed_d)
        out_embed_d = F.tanh(out_embed_d)
        out_embed_d = out_embed_d.permute(0,2,1,3)
        out_embed_d = self.embed_d_conv2(out_embed_d).squeeze()
        out_embed_d = F.tanh(out_embed_d)
        out_embed_d = out_embed_d.permute(2,1,0)
    
        out_distrib = out_embed_q*out_embed_d
        out_distrib = out_distrib.permute(2,1,0).contiguous()
        out_distrib = out_distrib.view(out_distrib.size(0),-1) 
        out_distrib = self.distrib_linear1(out_distrib)
        out_distrib = F.tanh(out_distrib)
        out_distrib = self.distrib_linear2(out_distrib)
        out_distrib = F.tanh(out_distrib)
        out_distrib = self.distrib_dropout(out_distrib)
        out_distrib = F.tanh(out_distrib)
        out_distrib = self.distrib_linear3(out_distrib)
        out_distrib = F.tanh(out_distrib)
        score = out_distrib + out_local
        return score.squeeze()
                        
