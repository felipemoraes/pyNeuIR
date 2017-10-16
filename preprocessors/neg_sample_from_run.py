"""Samples negative pairs from run."""

import argparse
from utils import load_qrels, load_run
import numpy as np

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-run')

    parser.add_argument('-qrel')

    parser.add_argument('-n', type=int)

    parser.add_argument('-top', type=int)

    parser.add_argument('-o')

    args = parser.parse_args()

    run, _ = load_run(args.run)

    qrels, _ = load_qrels(args.qrel)
   
    n = args.n

    top = args.top

    f = open(args.o, "w")

    np.random.seed(230)

    # Here documents with the lowest label (e.g, 0) should be ranked lower
    for qid in run:
        if qid not in qrels:
            continue
        # Get relevants for query
        rels = set()
        for label in qrels[qid]:
            if label != "0":
                for doc in qrels[qid][label]:
                    rels.add(doc)
        # Get top 100 non rel docs
        top_nonrels = [doc for doc in sorted(run[qid], key=run[qid].get, reverse=True) if doc not in rels][:top]
        for rel_doc in rels:
            sample_neg_docs = np.random.choice(top_nonrels, n, replace=False)
            sample_neg_docs = " ".join(sample_neg_docs)
            f.write("{} {} {}\n".format(qid, rel_doc, sample_neg_docs))
    f.close()

if __name__ == "__main__":
    main()
