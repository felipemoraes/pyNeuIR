import sys
import numpy as np
import json
from gensim.models.keyedvectors import KeyedVectors
# Fixed seed for reproducibility
np.random.seed(222)

class PreTrainedWordEmbeddings():

    def __init__(self, w2v_path, dim=300):
        self.w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        self.dim = dim
        self.oov = {}
        
    def __call__(self, word):
        if word in self.w2v_model.wv.vocab:
            return self.w2v_model.wv[word]
        elif word in self.oov:
            return self.oov[word]
        else:
            # It deals with OOV like described here: http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf
            self.oov[word] = np.random.rand(self.dim)
            return self.oov[word]

def matching_histogram_mapping(query_tvs, doc_tvs, num_bins):
    histograms = [histogram_mapping([np.cos(query_tv,doc_tv) for doc_tv in doc_tvs],num_bins) for query_tv in query_tvs]
    return histograms

def histogram_mapping(values, num_bins):
    count = [0.0]*num_bins
    for value in values:
        bin_idx = int(((value.data[0]+1)*(num_bins-1)) / 2)
        count[bin_idx] += 1
    return count

def nh(values):
    s = np.sum(values)
    if (s > 0):
        return [v/s for v in values]
    return values

def lnh(values):
    return [np.log(v) if v > 0 else 0 for v in values]


def load_qrels(qrels_path):
    qrels = {}
    docnos = set()
    for line in open(qrels_path):
        qid, _, docno, label = line.strip().split()
        if not qid in qrels:
            qrels[qid] = {}
        if not label in qrels[qid]:
            qrels[qid][label] = []
        qrels[qid][label].append(docno)
        docnos.add(docno)
    return qrels, docnos



def load_topics(topics_file, field_type):
    lines = [line.strip() for line in open(topics_file) if len(line.strip().replace(" ",""))> 0]
    topics = []
    for i, line in enumerate(lines):
        if "<num> " in line:
            qid = line.replace("<num> Number: ", "")
            topic = ""
            if field_type == "title":
                if not "<title> " in lines[i+1]:
                    topic = lines[i+2]
                else:
                    topic = lines[i+1].replace("<title> ", "")
            elif field_type == "desc":
                topic = lines[i+3]
            topics.append((qid, topic))
    return topics  


def escape(input):
    return input.translate({
        ord('('): None,
        ord(')'): None,
        ord('\''): None,
        ord('\"'): None,
        ord('.'): ' ',
        ord(':'): ' ',
        ord('\t'): ' ',
        ord('/'): ' ',
        ord('&'): ' ',
        ord(','): ' ',
        ord('-'): ' ',
        ord('?'): ' ',
        ord('+'): ' ',
        ord(';'): ' ',
        ord('`'): None,
        ord('$'): None,
        ord('<'): ' ',
        ord('>'): ' ',
        ord('%'): ' ',
    })

def generate_query_doc_pairs(qrels):
    f = open("training_pairs.txt", "w")
    for qid in qrels:
        label_list = sorted(qrels[qid].keys(), reverse = True)
        for hidx, high_label in enumerate(label_list[:-1]):
            for low_label in label_list[hidx+1:]:
                for high_d in qrels[qid][high_label]:
                    for low_d in qrels[qid][low_label]:
                        f.write("{} {} {}\n".format(qid,high_d,low_d))
    f.close()

