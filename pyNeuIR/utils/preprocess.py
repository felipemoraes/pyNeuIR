import operator
import numpy as np
import torch 
import json

def pad(tensor, length):
    return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

def pad_sequences(sequences,):
    max_len = max([len(sequence) for sequence in sequences])
    padded_sequences = [[]]*len(sequences)
    for i, sequence in enumerate(sequences):
        padded_sequences[i] = pad(sequence.clone(), max_len)
    return torch.stack(padded_sequences)
   
def load_histograms(histogram_file, max_len):
    histograms = {}
    for line in open(histogram_file):
        qid, docno, hs = line.split(" ", 2)
        hs = json.loads(hs)
        hs = pad(torch.FloatTensor(np.array(hs)),max_len)
        if not qid in histograms:
            histograms[qid] = {}
        histograms[qid][docno] = hs
    return histograms


def load_embeddings(queries_tvs_file, max_len):
    embeddings = {line.split(" ",1)[0]: json.loads(line.strip().split(" ",1)[1]) for line in open(queries_tvs_file)}
    for t in embeddings:
        embeddings[t] = pad(torch.FloatTensor(np.array(embeddings[t])),max_len)
    return embeddings


def load_idfs(queries_idfs_file, max_len):

    idfs = {line.split(" ", 1)[0]:  json.loads(line.strip().split(" ",1)[1]) for line in open(queries_idfs_file)}
    for t in idfs:
        idfs[t] = pad(torch.FloatTensor(np.array(idfs[t])),max_len).unsqueeze(1)
    return idfs

def to_binary_matrix(features, n_rows, n_cols):
    binary_matrix = np.zeros((n_rows,n_rows))
    for f in features:
        binary_matrix[f[0], f[1]] = 1.0
    return binary_matrix

def to_freq_matrix(features, n_rows, n_cols):
    freq_matrix = np.zeros((n_rows,n_rows))
    for f in features:
        freq_matrix[f[0], f[1]] = f[2]
    return freq_matrix

def load_features_local(features_local_file, n_rows, n_cols, train_file, validation_file):
    queries = [line.strip().split()[0] for line in open(train_file)]
    queries.extend([line.strip().split()[0] for line in open(validation_file)])
    queries = set(queries)
    features_local = {}
    for line in open(features_local_file):
        qid, docno, features = line.split(" ", 2)
        if qid in  queries:
            if not qid in features_local:
                features_local[qid] = {}
            features_local[qid][docno] = json.loads(features)
    return features_local

def load_features_distrib_query(features_distrib_file, n_rows, n_cols, train_file, validation_file):
    queries = [line.strip().split()[0] for line in open(train_file)]
    queries.extend([line.strip().split()[0] for line in open(validation_file)])
    queries = set(queries)

    features_dist = {}
    for line in open(features_distrib_file):
        qid, features = line.split(" ", 1)
        if qid in  queries:
            features_dist[qid] = json.loads(features)

    return features_dist

def load_features_distrib_doc(features_distrib_file, n_rows, n_cols, train_file, validation_file):
    docs = [line.strip().split()[1] for line in open(train_file)]
    docs.extend([line.strip().split()[1] for line in open(validation_file)])
    docs = set(docs)

    features_dist = {}
    for line in open(features_distrib_file):
        doc, features = line.split(" ", 1)
        if doc in docs:
            features_dist[qid][docno] = json.loads(features)
    return features_dist 
