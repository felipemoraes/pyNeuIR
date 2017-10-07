from collections import Counter
import sys
import pyndri
import numpy as np
import json
import glob

from input_utils import *

sys.path.append('../../pyNeuIR/')

from pyNeuIR.configs.interaction_featurizer_config import config


def letter_ngrams(word, n=5):
    ngrams = []
    word = "#" + word + "#"
    for n_ in range(1,n+1):
        for i in range(0,len(word)-n_+1):
            ngrams.append(word[i:i+n_])
    return ngrams


def docids_from_index(index, docnos):
    docids = {}
    for docid in range(index.document_base(), index.maximum_document()):
        docno, doc = index.document(docid)
        if docno in docnos:
            docids[docno] = docid
    return docids

def get_top_ngraph(index, top_ngraph=2000):
    token2id, id2token, id2df = index.get_dictionary()
    id2tf = index.get_term_frequencies()
    ngraph = Counter()
    for term_id in id2token:
        term = id2token[term_id]
        freq = id2tf[term_id]
        ngrams = letter_ngrams(term)
        for ngram in ngrams:
            ngraph[ngram] += freq
    
    return {x[0]:i for i, x in enumerate(ngraph.most_common(top_ngraph))}

def ngraph_counterizer(ngraph, terms):
    counter = []
    ngram_count = Counter()
    for i, term in enumerate(terms):
        ngrams = letter_ngrams(term)
        for ngram in ngrams:
            if ngram in ngraph:
                ngram_count[ngraph[ngram]] += 1
    return ngram_count

def load_training(training_set_file):
    queries = []
    docs = []
    for line in open(training_set_file):
        qid, doc = line.strip().split(" ", 1)
        queries.append(qid)
        docs.extend(doc.split())
    return set(queries), set(docs)
def main(argv):

    if len(config) < 4:
        print("Invalid configuration file.")
        sys.exit(0)
    
    training_set_file = argv[1]

    print("Loading topics")
    topics = load_topics(config["topics"], config["type"])
    topics = dict(topics)
    print("Loading qrels")
    qrels, docnos = load_training(training_set_file)
    print(len(docnos))
    

    print("Generating ngraph features")
    with pyndri.open(config["index"]) as index:
        token2id, id2token, id2df = index.get_dictionary()
        top_ngraphs = get_top_ngraph(index)
        f = open("top2k_ngraph.txt", "w")
        for top_ngraph in top_ngraphs:
            f.write("{} {}\n".format(top_ngraph, top_ngraphs[top_ngraph]))
        f.close()
        
        f = open("queries_term.txt", "w")
        for qid in qrels:
            query_matrix = []
            query = topics[qid]
            query_terms = [ term for term in pyndri.tokenize(escape(query.lower())) if term in token2id]
            ids = [token2id[term] for term in query_terms]
            f.write("{} {}\n".format(qid, " ".join(query_terms)))
        f.close()

        docids = docids_from_index(index, docnos)
        f = open("docs_term.txt", "w")
        for docno in docnos:
            docno, doc = index.document(docids[docno])
            doc = [w for w in doc if w >0]
            doc_terms = [id2token[w] for w in doc]
            f.write("{} {}\n".format(docno, " ".join(doc_terms)))
        f.close()

if __name__=='__main__':
    main(sys.argv)