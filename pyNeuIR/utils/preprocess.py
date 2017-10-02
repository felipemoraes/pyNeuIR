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