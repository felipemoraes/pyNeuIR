import sys
import os
import time
import numpy as np
import argparse

sys.path.append('.')

from models.duet import Duet
from configs.duet_config import config as c
from datasets.duet_dataset import DuetDataset
import torch
from torch.autograd import Variable
import torch.nn as nn
torch.manual_seed(222)

use_cuda = torch.cuda.device_count() > 0
        
if use_cuda:
    torch.cuda.manual_seed(222)



from torch.utils.data import Dataset
import numpy as np
np.random.seed(230)
import time

class TestDataset(Dataset):

    def __init__(self, test_instances, ngraphs = None,  model_type="duet", 
                max_query_words=10, max_doc_words=1000, num_ngraphs=2000):

        self.model_type = model_type

        self.max_query_words = max_query_words
        self.max_doc_words = max_doc_words
        self.num_ngraphs = num_ngraphs

        self.test_instances = [line.strip().split("\t") for line in test_instances]
        print("Loaded {} instances.".format(len(self.test_instances)))
        self.len = len(self.test_instances)

        if model_type == "duet" or model_type == "distrib":
            self.ngraphs = ngraphs
            print("Loaded term ngraphs")
        
       

    def get_features_local(self,query_terms, docs_terms):
        num_docs = len(docs_terms)
        features = np.zeros((num_docs, self.max_query_words, self.max_doc_words), dtype=np.float32)
        for doc_idx, doc in enumerate(docs_terms):
            for qw_idx, qword in enumerate(query_terms):
                if qw_idx == self.max_query_words:
                    break
                for dw_idx, dword in enumerate(doc):
                    if dw_idx == self.max_doc_words:
                        break
                    if qword == dword:
                        features[doc_idx, qw_idx, dw_idx] = 1
        return features
    
    def get_features_distrib_doc(self,docs_terms):
        num_docs = len(docs_terms)
        features = np.zeros((num_docs, self.num_ngraphs, self.max_doc_words), dtype=np.float32)
        for doc_idx, doc in enumerate(docs_terms):
            for dw_idx, dword in enumerate(doc):
                if dw_idx == self.max_doc_words:
                    break
                d_ngraphs = self.ngraphs[dword]
                for d_ngraph in d_ngraphs:
                    features[doc_idx, d_ngraph, dw_idx] += 1
        return features

    def get_features_distrib_query(self, query_terms):
        features = np.zeros((self.num_ngraphs, self.max_query_words), dtype=np.float32)
        for qw_idx, qword in enumerate(query_terms):
            if qw_idx == self.max_query_words:
                break
            q_ngraphs = self.ngraphs[qword]
            for q_ngraph in q_ngraphs:
                features[q_ngraph, qw_idx] += 1
        return features

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        item = {}
        qid, query_terms, docno, doc_terms = self.test_instances[idx]
        item["qid"] = qid
        item["docno"] = docno
        query_terms = list(map(int,v) for v in query_terms.split())
        doc_terms = list(map(int,v) for v in doc_terms.split())
        if self.model_type == "local" or self.model_type == "duet":
            item["features_local"] = self.get_features_local(query_terms, [doc_terms])
        if self.model_type == "distrib" or self.model_type == "duet":
            item["features_distrib_query"] = self.get_features_distrib_query(query_terms)
            item["features_distrib_doc"] = self.get_features_distrib_doc(query_terms, [doc_terms])
        return item

def to_cuda(variable):
    if use_cuda:
        return variable.cuda()
    return variable

def get_scores(dataloader):
    scores = []
    num_docs = 1
    for i, data in enumerate(dataloader, 0):
        time_start = time.time()

        # Initialize variables
        features_local = None
        features_distrib_query = None
        features_distrib_doc = None
        queries = data["queries"]
        docnos = data["docnos"]
        # Load variables according to the model type
        if model_type == "local" or model_type == "duet":
            features_local = data["features_local"]
            features_local = [to_cuda(Variable(features_local[:,i,:,:]).unsqueeze(1)) for i in range(num_docs)]

        if model_type == "distrib" or model_type == "duet":
            features_distrib_query = data["features_distrib_query"]
            features_distrib_doc = data["features_distrib_doc"]
            features_distrib_query = to_cuda(Variable(features_distrib_query).unsqueeze(1))
            features_distrib_doc = [to_cuda(Variable(features_distrib_doc[:,i,:,:]).unsqueeze(1)) for i in range(num_docs)]
        

        doc_scores = duet(features_local, features_distrib_query, features_distrib_doc, num_docs)
        doc_scores = num_docs.data.cpu().numpy()
        for i, score in enumerate(scores):
            scores.append((queries[i],docnos[i], score))
    return scores


def reranker(dataset_dir, save_dir, experiment_name, model_type="duet"):
    lines = [line.strip().split(" ", 1) for line in open(dataset_dir+"ngraphs.txt")]
    ngraphs = [[]]*(len(lines)+1)
    for line in lines:
    ngraphs[int(line[0])] = [int(v) for v in line[1].split()] 

    duet = Duet(c["n_q"],c["n_d"], c["m_d"], model_type)
    if use_cuda:
        duet = duet.cuda()

    t_start = time.time()

    f_baseline = open(dataset_dir+"test_data.txt")
    f_run = open(save_dir + experiment_name, "w")
    test_instances = []
    for line in f_baseline:
        if len(test_instances) == 50000:
            testset = DuetDataset(f_baseline, ngraphs, model_type)
            dataloader = torch.utils.data.DataLoader(testset, batch_size=8, num_workers=10)
            scores = get_scores(dataloader)
            for score in scores:
                f_run.write("{} Q0 {} 1 {} {}".format(score[0], score[1], score[2], experiment_name))
            test_instances = []
        test_instances.append(line)

    if len(test_instances) > 0:
        testset = DuetDataset(f_baseline, ngraphs, model_type)
        dataloader = torch.utils.data.DataLoader(testset, batch_size=8, num_workers=10)
        scores = get_scores(dataloader)
        for score in scores:
            f_run.write("{} Q0 {} 1 {} {}".format(score[0], score[1], score[2], experiment_name))
    f.close()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', help="Dataset folder")

    parser.add_argument('-test', help="Test queries")

    parser.add_argument('-o', help="Output folder")

    parser.add_argument('-name', help="Run name")

    parser.add_argument('-type', help="Type of network", choices=['duet', 'local', 'distrib'], default="duet")

    args = parser.parse_args()

    print("Loading dataset.")

    if not os.path.exists(args.o):
        os.makedirs(args.o)
    
    reranker(dataloader, args.o, args.name, args.type)
    
main()
