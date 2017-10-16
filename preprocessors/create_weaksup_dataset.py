from collections import Counter
import sys
import pyndri
import numpy as np
import json
import glob
import gzip
import re

from input_utils import *

sys.path.append('../../pyNeuIR/')

from pyNeuIR.configs.histograms_config import config

substrings = ["http", "www.", ".com", ".net", ".org", ".edu"]

def is_url_substring(query_string):
    for substring in substrings:
        if substring in query_string:
            return True
    return False

def main(argv):

    if len(argv) < 1:
        print("Invalid configuration file.")
        sys.exit(0)
    
    query_log_fold = argv[1]

    print("Generating candidate queries")
    candidate_queries = []
    pattern = re.compile('([^\s\w]|_)+')
    for query_log_file in glob.glob(query_log_fold + "user-ct-test-collection-*"):
        f = gzip.open(query_log_file)
        # skip first line
        f.readline()
        for line in f:
            line = line.decode("utf-8").split("\t")
            query_string = line[1]
            if is_url_substring(query_string):
                continue
            query_string = pattern.sub('', query_string)
            candidate_queries.append(query_string)
    candidate_queries = set(candidate_queries)
    print("Found {} candidate queries".format(len(candidate_queries)))


    print("Generating pseudo labels")
    f_query = open("training_query_set.txt", encoding='utf-8',  mode="w")
    f_label = open("training_pseudo_labels.txt", "w")
    with pyndri.open(config["index"]) as index:
        i = 0
        bm25_query_env = pyndri.OkapiQueryEnvironment(index,k1=1.2, b=0.75, k3=1000)
        for candidate_query in candidate_queries:
            try:
                results = index.query(candidate_query, results_requested=1000)
            except:
                print(candidate_query)
                continue
            if len(results) < 10:
                continue
            f_query.write("{} {}\n".format(i, candidate_query) )
            for docid, score in results:
                docno, _ = index.document(docid)
                f_label.write("{} {} {}\n".format(i,docno,score))
            i += 1
        f.close()
        print("Finished with {} queries".format(i))


if __name__=='__main__':
    main(sys.argv)

