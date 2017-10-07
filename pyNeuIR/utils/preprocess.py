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

def to_binary_matrix(query_terms, doc_terms, n_rows, n_cols):
    binary_matrix = np.zeros((n_rows,n_cols))
    for doc_term, doc_pos in doc_terms.items():
        if doc_pos >= n_rows:
            continue
        if doc_term in query_terms:
            query_pos = query_terms[doc_term]
            binary_matrix[doc_pos,query_pos] = 1
    return torch.FloatTensor(binary_matrix)

def letter_ngrams(word, n=5):
    ngrams = []
    word = "#" + word + "#"
    for n_ in range(1,n+1):
        for i in range(0,len(word)-n_+1):
            ngrams.append(word[i:i+n_])
    return ngrams

def to_freq_matrix(ngraphs, terms, n_rows, n_cols):
    freq_matrix = np.zeros((n_rows,n_cols))
    for doc_term, doc_pos in terms.items():
        ngrams = letter_ngrams(doc_term)
        if doc_pos >= n_cols:
            continue
        for ngram in ngrams:
            if ngram in ngraphs:
                freq_matrix[ngraphs[ngram],doc_pos] += 1
    return torch.FloatTensor(freq_matrix)

