import sys
import pyndri
import numpy as np
import json
import glob

from input_utils import *

sys.path.append('../../pyNeuIR/')

from pyNeuIR.configs.interaction_featurizer_config import config


def generate_query_sparse_matrix(token2id, query):
    sparse_matrix = {}
    query_terms = pyndri.tokenize(escape(query.lower()))
    for i, query_term in enumerate(query_terms):
        if query_term in token2id:
            term_id = token2id[query_term]
            sparse_matrix[term_id] = i
    return sparse_matrix

def generate_doc_sparse_matrix(index, docid):
    docno, doc = index.document(docid)
    return doc[:1000]

def docids_from_index(index, docnos):
    docids = {}
    for docid in range(index.document_base(), index.maximum_document()):
        docno, doc = index.document(docid)
        if docno in docnos:
            docids[docno] = docid
    return docids

def generate_binary_matrix(query_matrix, doc_matrix):
    binary_matrix = []
    for doc_pos, doc_term in enumerate(doc_matrix):
        if doc_term in query_matrix:
            query_pos = query_matrix[doc_term]
            binary_matrix.append((doc_pos,query_pos))
    return binary_matrix

def main(argv):

    if len(config) < 4:
        print("Invalid configuration file.")
        sys.exit(0)

    print("Loading topics")
    topics = load_topics(config["topics"], config["type"])
    topics = dict(topics)
    print("Loading qrels")
    qrels, docnos = load_qrels(config["qrels"])

    print("Generating local features")
    f = open("features_local.txt", "w")
    with pyndri.open(config["index"]) as index:
        token2id, id2token, id2df = index.get_dictionary()
        docids = docids_from_index(index, docnos)
        for qid in qrels:
            query_matrix = generate_query_sparse_matrix(token2id, topics[qid])
            for label in qrels[qid]:
                for docno in qrels[qid][label]:
                    docid = docids[docno]
                    doc_matrix = generate_doc_sparse_matrix(index, docid)
                    binary_matrix = generate_binary_matrix(query_matrix, doc_matrix)
                    f.write("{} {} {}\n".format(qid,docno,json.dumps(binary_matrix)))
    
    f.close()


    


if __name__=='__main__':
    main(sys.argv)