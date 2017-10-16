import sys
# import numpy as np
import json
# from gensim.models.keyedvectors import KeyedVectors
# # Fixed seed for reproducibility
# np.random.seed(222)

# class PreTrainedWordEmbeddings():

#     def __init__(self, w2v_path, dim=300):
#         self.w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
#         self.dim = dim
#         self.oov = {}
        
#     def __call__(self, word):
#         if word in self.w2v_model.wv.vocab:
#             return self.w2v_model.wv[word]
#         elif word in self.oov:
#             return self.oov[word]
#         else:
#             # It deals with OOV like described here: http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf
#             self.oov[word] = np.random.rand(self.dim)
#             return self.oov[word]

def compute_similarity(A_vec, B_vec):
    dot = np.dot(A_vec,B_vec)
    normA = np.linalg.norm(A_vec)
    normB = np.linalg.norm(B_vec)
    sim = dot/(normA*normB)
    return sim

def matching_histogram_mapping(query_tvs, doc_tvs, num_bins):
    histograms = [histogram_mapping([compute_similarity(query_tv,doc_tv) for doc_tv in doc_tvs],num_bins) for query_tv in query_tvs]
    return histograms

def histogram_mapping(similarities, num_bins):
    count = [0.0]*num_bins
    for sim in similarities:
        bin_idx = int((sim + 1.0 ) / 2.0 * (num_bins - 1))
        count[bin_idx] += 1
    return count

def nh(values):
    s = np.sum(values)
    if (s > 0):
        return [v/s for v in values]
    return values

def lnh(values):
    return [np.log10(v+1) for v in values]


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

def load_run(run_path):
    run = {}
    docnos = set()
    for line in open(run_path):
        qid, _, docno, _, score, label = line.strip().split()
        if not qid in run:
            run[qid] = {}
        run[qid][docno] = float(score)
        docnos.add(docno)
    return run, docnos


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


# from graphviz import Digraph
# import torch
# from torch.autograd import Variable

# def make_dot(var, params=None):
#     """ Produces Graphviz representation of PyTorch autograd graph
#     Blue nodes are the Variables that require grad, orange are Tensors
#     saved for backward in torch.autograd.Function
#     Args:
#         var: output Variable
#         params: dict of (name, Variable) to add names to node that
#             require grad (TODO: make optional)
#     """
#     if params is not None:
#         assert isinstance(params.values()[0], Variable)
#         param_map = {id(v): k for k, v in params.items()}

#     node_attr = dict(style='filled',
#                      shape='box',
#                      align='left',
#                      fontsize='12',
#                      ranksep='0.1',
#                      height='0.2')
#     dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
#     seen = set()

#     def size_to_str(size):
#         return '('+(', ').join(['%d' % v for v in size])+')'

#     def add_nodes(var):
#         if var not in seen:
#             if torch.is_tensor(var):
#                 dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
#             elif hasattr(var, 'variable'):
#                 u = var.variable
#                 name = param_map[id(u)] if params is not None else ''
#                 node_name = '%s\n %s' % (name, size_to_str(u.size()))
#                 dot.node(str(id(var)), node_name, fillcolor='lightblue')
#             else:
#                 dot.node(str(id(var)), str(type(var).__name__))
#             seen.add(var)
#             if hasattr(var, 'next_functions'):
#                 for u in var.next_functions:
#                     if u[0] is not None:
#                         dot.edge(str(id(u[0])), str(id(var)))
#                         add_nodes(u[0])
#             if hasattr(var, 'saved_tensors'):
#                 for t in var.saved_tensors:
#                     dot.edge(str(id(t)), str(id(var)))
#                     add_nodes(t)
#     add_nodes(var.grad_fn)
#     return dot
