import sys
import pyndri
from preprocess import PreTrainedWordEmbeddings
import numpy as np
import json

if len(sys.argv) < 5:
    print("error")
    sys.exit(0)

def load_qrels(qrels_path):
    qrels = {}
    for line in open(qrels_path):
        qid, _, docno, label = line.strip().split()
        qid = int(qid)
        if not qid in qrels:
            qrels[qid] = []
        qrels[qid].append(docno)
    return qrels


qrels = load_qrels(sys.argv[1])

topics = {}
for line in open(sys.argv[3]):
    tid, terms = line.strip().split("\t")
    terms = [ int(x) for x in terms.split()]
    tid = int(tid)
    topics[tid] = terms


docs = {}
for line in open(sys.argv[4]):
    docno, docid = line.strip().split("\t")
    docs[docno] = int(docid)

embeddings = PreTrainedWordEmbeddings(sys.argv[5])

def matching_histogram_mapping(query_tvs, doc_tvs, num_bins):
    # Local interaction
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


with pyndri.open(sys.argv[2]) as index:
    token2id, id2token, id2df = index.get_dictionary()

    f_qtvs = open("queries_tv.txt", "w")
    f_ch = open("histograms_ch.txt", "w")

    f_nh = open("histograms_nh.txt", "w")

    f_lch = open("histograms_lch.txt", "w")
    c = 0
    for topic in qrels:
        query_tvs = [embeddings(id2token[w]) for w in topics[topic] if w > 0]
        f_qtvs.write("{} {}\n".format(topic, json.dumps([q.tolist() for q in query_tvs]) ))
        for docno in qrels[topic]:
            document_id = docs[docno]
            docno, doc = index.document(document_id)
            doc_tvs = [embeddings(id2token[w]) for w in doc if w > 0 ]          
            histograms = matching_histogram_mapping(query_tvs, doc_tvs, 30)
            f_ch.write("{} {} ".format(topic,docno))
            f_nh.write("{} {} ".format(topic,docno))
            f_lch.write("{} {} ".format(topic,docno))
            for histogram in histograms[:-1]:
                f_ch.write(" ".join(map(str,histogram)) + "\t")
                f_nh.write(" ".join(map(str,nh(histogram))) + "\t")
                f_lch.write(" ".join(map(str,lnh(histogram))) + "\t")
            f_ch.write(" ".join(map(str,histogram)) + "\n")
            f_nh.write(" ".join(map(str,nh(histogram)))+ "\n")
            f_lch.write(" ".join(map(str,lnh(histogram)))+ "\n")
        c += 1
        print("Processed {} queries.".format(c))
            
    f_ch.close()
    f_nh.close()
    f_lch.close()
    f_qtvs.close()