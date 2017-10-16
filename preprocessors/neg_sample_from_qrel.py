"""Samples negative pairs from qrel."""

import argparse
from utils import load_qrels
import numpy as np

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-qrel')

    parser.add_argument('-n', type=int)

    parser.add_argument('-o')

    args = parser.parse_args()

    qrels, _ = load_qrels(args.qrel)
   
    n = args.n

    f = open(args.o, "w")

    np.random.seed(230)

    # Here documents with the lowest label (e.g, 0) should be ranked lower
    for qid in qrels:
        label_list = sorted(qrels[qid].keys(), reverse = True)
        lowest_label = label_list[-1]
        for hidx, high_label in enumerate(label_list[:-1]):
            for rel_doc in qrels[qid][high_label]:
                if len(qrels[qid][lowest_label]) < n:
                    continue
                sample_neg_docs = np.random.choice(qrels[qid][lowest_label], n, replace=False)
                sample_neg_docs = " ".join(sample_neg_docs)
                f.write("{} {} {}\n".format(qid, rel_doc, sample_neg_docs))
    f.close()

if __name__ == "__main__":
    main()
