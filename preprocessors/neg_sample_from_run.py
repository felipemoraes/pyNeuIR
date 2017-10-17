"""Samples negative pairs from run."""

import argparse
from utils import load_qrels, load_run
import numpy as np

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-run')

    parser.add_argument('-qrel')

    parser.add_argument('-p', type=int)

    parser.add_argument('-n', type=int)

    parser.add_argument('-top', type=int)

    parser.add_argument('-o')

    args = parser.parse_args()

    qrels, _ = load_qrels(args.qrel)
   
    n = args.n

    p = args.p

    top = args.top

    f = open(args.o, "w")

    np.random.seed(230)

    # Here documents with the lowest label (e.g, 0) should be ranked lower
    previous_qid = "-"
    for line in open(args.run):
        qid, _, doc, _, score, _ = line.strip().split()
        if qid not in qrels:
            continue
        # Get relevants for query
        rels = set()
        if qid != previous_qid and previous_qid != "-" :
            
            for label in qrels[qid]:
                if label != "0":
                    for doc in qrels[qid][label]:
                        rels.add(doc)
            # Get top 100 non rel docs
            top_nonrels = [doc for doc in sorted(results, key=results.get, reverse=True) if doc not in rels][:top]
            rels = list(rels)
            if len(rels) > p:
                rels = np.random.choice(rels, p, replace=False)
            for rel_doc in rels:
                if len(qrels[qid][lowest_label]) < n:
                    sample_neg_docs = top_nonrels
                    r = n - len(sample_neg_docs)
                    for i in range(r):
                        sample_neg_docs.append(np.random.choice(top_nonrels, 1))
                else:
                    sample_neg_docs = np.random.choice(top_nonrels, n, replace=False)

                sample_neg_docs = " ".join(sample_neg_docs)
                f.write("{} {} {}\n".format(qid, rel_doc, sample_neg_docs))
            results = {doc: float(score)}
        elif previous_qid == "-":
            results = {doc: float(score)}
        else:
            results[doc] = float(score)
        previous_qid = qid
    f.close()

if __name__ == "__main__":
    main()
