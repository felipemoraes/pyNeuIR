import sys
import pyndri
import math
def load_topics(topics_file, field_type):
    lines = [line.strip() for line in open(topics_file) if len(line.strip().replace(" ",""))> 0]
    topics = []
    for i, line in enumerate(lines):
        if "<num> " in line:
            qid = line.replace("<num> Number: ", "")
            topic = ""
            if field_type == "title":
                if not "<title> " in lines[i+1]:
                    topic = lines[i+2]
                else:
                    topic = lines[i+1].replace("<title> ", "")
            elif field_type == "desc":
                topic = lines[i+3]
            topics.append((qid, topic))
    return topics  


def escape(input):
    return input.translate({
        ord('('): None,
        ord(')'): None,
        ord('\''): None,
        ord('\"'): None,
        ord('.'): ' ',
        ord(':'): ' ',
        ord('\t'): ' ',
        ord('/'): ' ',
        ord('&'): ' ',
        ord(','): ' ',
        ord('-'): ' ',
        ord('?'): ' ',
        ord('+'): ' ',
        ord(';'): ' ',
        ord('`'): None,
        ord('$'): None,
        ord('<'): ' ',
        ord('>'): ' ',
        ord('%'): ' ',
    })

def load_qrels(qrels_path):
    qrels = {}
    docnos = set()
    for line in open(qrels_path):
        qid, _, docno, label = line.strip().split()
        if not qid in qrels:
            qrels[qid] = {}
        if not label in qrels[qid]:
            qrels[qid][label] = []
        qrels[qid][label].append(docno)
        docnos.add(docno)
    return qrels, docnos


qrels, docnos = load_qrels(sys.argv[1]) 

pair_list = []

f = open("training_pairs.txt", "w")

for tid in qrels:
    label_list = sorted(qrels[tid].keys(), reverse = True)
    for hidx, high_label in enumerate(label_list[:-1]):
        for low_label in label_list[hidx+1:]:
            for high_d in qrels[tid][high_label]:
                for low_d in qrels[tid][low_label]:
                    f.write("{} {} {}\n".format(tid, high_d,  low_d))
f.close()

topics = load_topics(sys.argv[3], sys.argv[4])

with pyndri.open(sys.argv[2]) as index:
    token2id, id2token, id2df = index.get_dictionary()
    id2tf = index.get_term_frequencies()
    f_id = open("queries_terms_id.txt", "w")
    f_t = open("queries_terms.txt", "w")
    f_idfs = open("/Developer/data/idfs.txt", "w")

    collection_size = float(index.document_count())
    for topic in topics:
        tid, query = topic
        query_terms = pyndri.tokenize(escape(query.lower()))
        tokens = [token2id[token] if token in token2id else 0 for token in query_terms]
        idfs = [math.log(collection_size/id2tf[token]) if token > 0 else 0 for token in tokens]
        f_idfs.write(tid +"\t" + " ".join(map(str,idfs)) + "\n")
        f_id.write(tid +"\t" + " ".join(map(str,tokens)) + "\n")
        f_t.write(tid +"\t" + " ".join(map(str,query_terms)) + "\n")
    f_id.close()
    f_t.close()
    f_idfs.close()
    f = open("docs_terms.txt", "w")
    for document_id in range(index.document_base(), index.maximum_document()):
        docno, doc = index.document(document_id)
        if docno in docnos:
            f.write("{0}\t{1}\n".format(docno , document_id))
    f.close()
    
