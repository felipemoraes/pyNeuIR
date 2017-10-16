"""Creates duet dataset."""

import argparse
import pyndri
import ujson as json
import os
from collections import Counter
from utils import escape

def get_top_ngraphs(ngraphs, max_ngraph_len, word):
    word_ngraphs = []
    token = '#' + word + '#'
    token_len = len(token)
    for i in range(token_len):
        for j in range(0, max_ngraph_len):
            if i+j < token_len:
                ngraph_idx = ngraphs.get(token[i:i+j])
                if ngraph_idx != None:
                    word_ngraphs.append(str(ngraph_idx))
    return word_ngraphs 

def get_token_ngraphs(token, max_ngraph_len=5):
    token = token.replace(" ", "")
    token = '#' + token + '#'
    token_len = len(token)
    token_ngraphs = []
    for i in range(token_len):
        for j in range(0, max_ngraph_len):
            if i+j < token_len:
                token_ngraphs.append(token[i:i+j])
    return token_ngraphs

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
    ngraphs = Counter()
    for term_id in id2token:
        term = id2token[term_id]
        freq = id2tf[term_id]
        ngrams = get_token_ngraphs(term,5)
        for ngram in ngrams:
            ngraphs[ngram] += freq
    
    return {x[0]:i for i, x in enumerate(ngraphs.most_common(top_ngraph))}


def get_docs_train(train_file):
    docs = []
    for line in open(train_file):
        qid, doc = line.strip().split(" ", 1)
        docs.extend(doc.split())
    return set(docs)


def get_docs_test(test_file):
    docs = set()
    # 684 Q0 LA102589-0032 1 -5.08734 indri
    for line in open(test_file):
        qid, _, doc, rank, score, label = line.strip().split()
        docs.add(doc)
    return docs

def load_queries(queries_file):
    queries = {}
    for line in open(queries_file):
        qid, query = line.split(" ", 1)
        queries[qid] = query.strip()
    return queries
    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-pairs')

    parser.add_argument('-run')

    parser.add_argument('-queries')

    parser.add_argument('-index')

    parser.add_argument('-o')

    args = parser.parse_args()

    if args.pairs:
        docs_train = get_docs_train(args.pairs)

    if args.run:
        docs_test = get_docs_test(args.run)

    queries = load_queries(args.queries)

    if not os.path.exists(args.o):
        os.makedirs(args.o)

    max_ngraph_len = 5

    
    with pyndri.open(args.index) as index:
        token2id, id2token, id2df = index.get_dictionary()
        top_ngraphs = get_top_ngraph(index)
        with open(args.o + "/ngraphs.txt", "w") as f:
            for term_id in id2token:
                term = id2token[term_id]
                ngraphs = get_top_ngraphs(top_ngraphs,5,term)
                f.write("{} {}\n".format(str(term_id), " ".join(ngraphs)))
            f.close()

        with open(args.o + "/terms.txt", "w") as f:
            for term_id in id2token:
                term = id2token[term_id]
                f.write("{} {}\n".format(str(term_id), term))
            f.close()

        print("Got top ngraphs.")
        queries_data = {}
        docs_data = {}

        def get_query_obj(qid):
            query = queries[qid]
            query_terms = [term for term in pyndri.tokenize(escape(query.lower())) if term in token2id]
            ids = [str(token2id[term]) for term in query_terms]
            query_obj = " ".join(ids)
            return query_obj

        def get_doc_obj(docids,docno):
            docno, doc = index.document(docids[docno])
            doc = [str(w) for w in doc if w >0][:1000]
            doc_obj =  " ".join(doc)
            return doc_obj

        if args.pairs:

            with open(args.o + "/train_data.txt", "w") as f:

                print("Processing train...")

                docids = docids_from_index(index, docs_train)

                for line in open(args.pairs):
                    qid, docs = line.strip().split(" ", 1)
                    docs = docs.split()
                    query_obj = get_query_obj(qid)
                    f.write("{}\t{}\t{}\n".format(qid, query_obj, "\t".join([get_doc_obj(docids,d) for d in docs])))
                f.close()
                    

        if args.run:
        
            with open(args.o + "/test_data.txt", "w") as f:
                docids = docids_from_index(index, docs_test)
                 
                print("Processing test...")

                for line in open(args.run):
                    qid, _, doc, rank, score, label = line.strip().split()
                    query_obj = get_query_obj(qid)
                    doc_obj = get_doc_obj(docids,doc)
                    f.write("{}\t{}\t{}\t{}\n".format(qid, query_obj ,doc, doc_obj))
                f.close()
       

if __name__ == "__main__":
    main()