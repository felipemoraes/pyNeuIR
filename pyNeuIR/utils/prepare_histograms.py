import sys
import pyndri
import numpy as np
import json
import glob

from input_utils import *

sys.path.append('../../pyNeuIR/')

from pyNeuIR.configs.histograms_config import config


def generate_queries_tvs(index_path, topics, embeddings):

    f_qtvs = open("queries_tvs.txt", "w")
    f_idfs = open("queries_idfs.txt", "w")
    queries_tvs = {}

    with pyndri.open(index_path) as index:
        token2id, id2token, id2df = index.get_dictionary()
        id2tf = index.get_term_frequencies()

        collection_size = float(index.document_count())
        for topic in topics:
            qid, query = topic
            query_terms = pyndri.tokenize(escape(query.lower()))
            query_tvs = [embeddings(w) for w in query_terms if w in token2id]
            queries_tvs[qid] = query_tvs
            
            tokens = [token2id[term] for term in query_terms if term in token2id]
            idfs = [np.log(collection_size/id2tf[token]) if token > 0 else 0 for token in tokens]

            f_qtvs.write("{} {}\n".format(qid, json.dumps([tv.tolist() for tv in query_tvs]) ))
            f_idfs.write("{} {}\n".format(qid, json.dumps(idfs) ))
    return queries_tvs

def docids_from_index(index_path, docnos):
    docids = {}
    with pyndri.open(index_path) as index:
        for docid in range(index.document_base(), index.maximum_document()):
            docno, doc = index.document(docid)
            if docno in docnos:
                docids[docno] = docid
    return docids

def generate_doc_tvs(index, docids, embeddings):
    docs_tvs = {}
    with pyndri.open(index_path) as index:
        token2id, id2token, id2df = index.get_dictionary()
        for docno in docids:
            docno, doc = index.document(docids[docno])
            doc_tvs = [embeddings(id2token[w]) for w in doc if w > 0]
            docs_tvs[docno] = doc_tvs
    return docs_tvs


def print_histograms(index_path, queries_tvs, docnos, queries_docs, embeddings, name="histogram"):
    
    f_ch = open(name + "_ch.txt", "w")
    f_nh = open(name + "_nh.txt", "w")
    f_lch = open(name + "_lch.txt", "w")
    
    docids = docids_from_index(index_path, docnos)
    count = 0
    for qid in queries_docs:
        count += 1
        docids_qid = {}
        query_tvs = queries_tvs[qid]

        for label in queries_docs[qid]:
            for docno in queries_docs[qid][label]:
                docids_qid[docno] = docids[docno] 
        
        with pyndri.open(index_path) as index:
            token2id, id2token, id2df = index.get_dictionary()
            for docno in docids_qid:
                docno, doc = index.document(docids[docno])
                doc_tvs = [embeddings(id2token[w]) for w in doc if w > 0]
            
                
    
                histograms = matching_histogram_mapping(query_tvs, doc_tvs, 30)
                
                f_ch.write("{} {} {}\n".format(qid,docno, json.dumps(histograms)))
                f_nh.write("{} {} {}\n".format(qid,docno, json.dumps([nh(histogram) for histogram in histograms ] )))
                f_lch.write("{} {} {}\n".format(qid,docno,json.dumps([lnh(histogram) for histogram in histograms ])))
        
        if count % 10 == 0:
            print("Processed {} queries.".format(count))
def load_test(test_file):
    run = {}
    docnos = set()
    # 684 Q0 LA102589-0032 1 -5.08734 indri
    for line in open(test_file):
        qid, _, docno, rank, score, tag = line.strip().split()
        if not qid in run:
            run[qid] = {}
            run[qid]["0"] = {}
        run[qid]["0"][docno] = 1
        docnos.add(docno)
    return run, docnos
    
def main(argv):

    if len(config) < 6:
        print("Invalid configuration file.")
        sys.exit(0)

    print("Loading topics")
    topics = load_topics(config["topics"], config["type"])
    print("Loading qrels")
    qrels, docnos = load_qrels(config["qrels"])
    print("Loading embeddings")
    
    embeddings = PreTrainedWordEmbeddings(config["embeddings"])

    print("Generating queries tvs")
    queries_tvs = generate_queries_tvs(config["index"], topics, embeddings)

    print("Printing histograms")
    print_histograms(config["index"], queries_tvs, docnos, qrels, embeddings)

    for test_file in glob.glob(config["tests_fold"]+"*"):
        file_name = test_file.split("/")[-1]
        run, docnos = load_test(test_file)
        print("Generating histograms for file {}.".format(file_name))
        print_histograms(config["index"], queries_tvs, docnos, run, embeddings, file_name)
    
    print("Generating query doc pairs for training")
    query_doc_pairs = generate_query_doc_pairs(qrels)

if __name__=='__main__':
    main(sys.argv)
