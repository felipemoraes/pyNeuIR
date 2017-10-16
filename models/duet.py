import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from .duet_local import DuetLocal
from .duet_distrib import DuetDistrib

class Duet(nn.Module):

    def __init__(self, words_per_query, words_per_doc, num_ngraphs, model_type = "duet"):
        super(Duet, self).__init__()
        
        self.model_type = model_type
        if model_type == "local" or model_type == "duet":
            self.net_local = DuetLocal(words_per_query, words_per_doc)
        if model_type == "distrib" or model_type == "duet":
            self.net_distrib = DuetDistrib(num_ngraphs, words_per_query, words_per_doc)        
        
    def forward(self, features_local, features_distrib_query, features_distrib_doc, num_docs):  
        
        if self.model_type == "duet":
            score = self.net_local(features_local,num_docs) + self.net_distrib(features_distrib_query, features_distrib_doc, num_docs)
        elif self.model_type == "local":
            score = self.net_local(features_local,num_docs)
        elif self.model_type == "distrib":
            score = self.net_distrib(features_distrib_query, features_distrib_doc, num_docs)
        return score   
                        
