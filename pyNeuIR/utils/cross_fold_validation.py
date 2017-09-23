import sys
import logging
import random


random.seed(8703)

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
            topics.append(qid + " " + topic + "\n")
    return topics  

topics_file = sys.argv[1]
save_path = sys.argv[2]
field_type = sys.argv[4]
n_folds = int(sys.argv[3])

topics =  load_topics(topics_file, field_type)

random.shuffle(topics)

len_t = len(topics)


fold_size = int(len_t/n_folds)

for i in range(n_folds):
    f = open(save_path + "test{}.txt".format(i),"w")
    for topic in topics[i*fold_size: i*fold_size+fold_size]:
        f.write(topic)
    f.close()

    f = open(save_path + "train{}.txt".format(i),"w")
    for topic in topics[0: i*fold_size]:
        f.write(topic)
    for topic in topics[i*fold_size+fold_size:]:
        f.write(topic)
    f.close()


