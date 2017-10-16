import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as weight_init
import torch.nn.functional as F

class DuetDistrib(nn.Module):

    def __init__(self, num_ngraphs, words_per_query, words_per_doc):
        super(DuetDistrib, self).__init__()
        
        self.num_hidden_nodes = 300
        self.word_window_size = 3
        self.pooling_kernel_width_query = words_per_query - self.word_window_size + 1 # = 8
        self.pooling_kernel_width_doc = 100
        self.num_pooling_windows_doc = ((words_per_doc - self.word_window_size + 1) - self.pooling_kernel_width_doc) + 1 # = 899

        self.embed_q_conv =  nn.Conv2d(1,self.num_hidden_nodes, (num_ngraphs, self.word_window_size), stride=(1,1))
        self.embed_q_max_pool  = nn.MaxPool2d((1, self.pooling_kernel_width_query), stride=(1, 1))
        self.embed_q_linear = nn.Linear(self.num_hidden_nodes, self.num_hidden_nodes)  

        self.embed_d_conv1 =  nn.Conv2d(1,self.num_hidden_nodes, (num_ngraphs, self.word_window_size), stride=(1,1))
        self.embed_d_max_pool  = nn.MaxPool2d((1, self.pooling_kernel_width_doc), stride=(1, 1))
        self.embed_d_conv2 =  nn.Conv2d(1,self.num_hidden_nodes, (self.num_hidden_nodes,1), stride=(1,1))

        self.distrib_linear1 = nn.Linear(self.num_hidden_nodes*self.num_pooling_windows_doc, self.num_hidden_nodes)
        self.distrib_linear2 = nn.Linear(self.num_hidden_nodes, self.num_hidden_nodes)
        self.distrib_dropout = nn.Dropout(p=0.2)  
        self.distrib_linear3 = nn.Linear(self.num_hidden_nodes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.xavier_uniform(m.weight.data, gain=1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                weight_init.xavier_uniform(m.weight.data, gain=1)
                m.bias.data.zero_()
        
    def forward(self, features_distrib_query, features_distrib_doc, num_docs):

        queries_embeddings = self.forward_queries_embeddings(features_distrib_query)
        docs_embeddings = [self.forward_doc_embeddings(features_distrib) for features_distrib in features_distrib_doc]
        scores = [self.forward_once(queries_embeddings, docs_embeddings[i]) for i in range(num_docs)]
        scores = torch.stack(scores).squeeze()
        if len(scores.size()) > 1:
            return scores.permute(1,0)
        return scores
    
    def forward_queries_embeddings(self, features_distrib_query):
        out_embed_q = self.embed_q_conv(features_distrib_query)
        out_embed_q = F.tanh(out_embed_q)
        out_embed_q = self.embed_q_max_pool(out_embed_q).squeeze()
        out_embed_q = F.tanh(out_embed_q)
        out_embed_q = self.embed_q_linear(out_embed_q)
        out_embed_q = F.tanh(out_embed_q)
        return out_embed_q
    
    def forward_doc_embeddings(self, features_distrib_doc):
        out_embed_d = self.embed_d_conv1(features_distrib_doc)
        out_embed_d = F.tanh(out_embed_d)
        out_embed_d = self.embed_d_max_pool(out_embed_d)
        out_embed_d = F.tanh(out_embed_d)
        out_embed_d = out_embed_d.squeeze()
        out_embed_d = self.embed_d_conv2(out_embed_d.unsqueeze(1)).squeeze()
        out_embed_d = F.tanh(out_embed_d).squeeze()
        out_embed_d = out_embed_d.permute(0,2,1)
        return out_embed_d
    
    def forward_once(self, out_embed_q, out_embed_d):
        out_distrib = out_embed_q.permute(1,0)*out_embed_d.permute(1,2,0)
        out_distrib = out_distrib.permute(2,1,0).contiguous()
        out_distrib = out_distrib.view(out_distrib.size(0),-1) 
        out_distrib = self.distrib_linear1(out_distrib)
        out_distrib = F.tanh(out_distrib)
        out_distrib = self.distrib_linear2(out_distrib)
        out_distrib = F.tanh(out_distrib)
        out_distrib = self.distrib_dropout(out_distrib)
        out_distrib = self.distrib_linear3(out_distrib)
        out_distrib = F.tanh(out_distrib)

        score = out_distrib

        return out_distrib
