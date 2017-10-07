import sys
import numpy as np
import json

from input_utils import *
    
def main(argv):
    
    if len(argv) < 1:
        print("Invalid configuration file.")
        sys.exit(0)

    print("Loading qrels")
    qrels, docnos = load_qrels(argv[1])

    print("Generating training set")

    f = open("training_set.txt", "w")
    
    np.random.seed(230)

    for qid in qrels:
        label_list = sorted(qrels[qid].keys(), reverse = True)
        for hidx, high_label in enumerate(label_list[:-1]):
            for low_label in label_list[hidx+1:]:
                for high_d in qrels[qid][high_label]:
                    if len(qrels[qid][low_label]) < 4:
                        print(qid, high_label, low_label, len(qrels[qid][high_label]),  len(qrels[qid][low_label]))
                        continue
                    sample_low_d = np.random.choice(qrels[qid][low_label], 4, replace=False)
                    sample_low_d = " ".join(sample_low_d)
                    
                    f.write("{} {} {}\n".format(qid,high_d,sample_low_d))
    f.close()
    
if __name__=='__main__':
    main(sys.argv)