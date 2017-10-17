from torch.utils.data import Dataset
import numpy as np
np.random.seed(230)
import time

class DuetDataset(Dataset):

    def __init__(self, dataset_folder, train_file, model_type="duet", 
            max_query_words=10, max_doc_words=1000, num_ngraphs=2000):

        self.model_type = model_type

        self.max_query_words = max_query_words
        self.max_doc_words = max_doc_words
        self.num_ngraphs = num_ngraphs

        self.train_queries = set([line.split(" ",1)[0] for line in open(train_file)])
        print("Loaded queries")
        
        self.train_instances = [line.strip().split("\t") for line in open(dataset_folder+"train_data.txt")]

        self.train_instances = [[ list(map(int, vec.split())) for vec in instance[1:]] for instance in self.train_instances if instance[0] in self.train_queries]

        print("Loaded {} instances.".format(len(self.train_instances)))
        np.random.shuffle(self.train_instances)
    
        lines = [line.strip().split(" ", 1) for line in open(dataset_folder+"ngraphs.txt")]
        self.ngraphs = [[]]*(len(lines)+1)
        for line in lines:
            self.ngraphs[int(line[0])] = [int(v) for v in line[1].split()] 
        
        self.len = len(self.train_instances)
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
        if self.model_type == "local" or self.model_type == "duet":
            item["features_local"] = self.get_features_local(self.train_instances[idx][0], self.train_instances[idx][1:])
        if self.model_type == "distrib" or self.model_type == "duet":
            item["features_distrib_query"] = self.get_features_distrib_query(self.train_instances[idx][0])
            item["features_distrib_doc"] = self.get_features_distrib_doc(self.train_instances[idx][1:])
        item["labels"] = 0
        return item
        
        
