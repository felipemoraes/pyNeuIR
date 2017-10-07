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
    for i, term in enumerate(terms):
        ngrams = letter_ngrams(term)
        ngram_count = Counter()
        for ngram in ngrams:
            if ngram in ngraph:
                ngram_count[ngram] += 1
        for ngram in ngram_count:
            counter.append((ngraph[ngram], i, ngram_count[ngram]))
    return counter

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
        docids = docids_from_index(index, docnos)
        f = open("query_ngraph_featurizer.txt", "w")
        for qid in qrels:
            query_matrix = []
            query = topics[qid]
            query_terms = pyndri.tokenize(escape(query.lower()))
            query_ngraph = ngraph_counterizer(top_ngraphs,query_terms)
            f.write("{} {}\n".format(qid, json.dumps(query_ngraph)))
        f.close()
        f = open("doc_ngraph_featurizer.txt", "w")
        for docno in docnos:
            docno, doc = index.document(docids[docno])
            doc_terms = [ id2token[w] for w in doc if w > 0]
            doc_ngraph = ngraph_counterizer(top_ngraphs,doc_terms)
            f.write("{} {}\n".format(docno, json.dumps(doc_ngraph)))
        f.close()

if __name__=='__main__':
    main(sys.argv)