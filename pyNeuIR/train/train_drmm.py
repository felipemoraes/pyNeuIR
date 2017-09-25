import sys
import os
import logging
import time
import json
import numpy as np

sys.path.append('../../pyNeuIR/')

from pyNeuIR.utils.preprocess import process_minibatch, PreTrainedWordEmbeddings, load_histograms
from pyNeuIR.utils.pairs_generator import PairsGenerator
from pyNeuIR.models.drmm import DRMM_TV, HingeLoss
from pyNeuIR.configs.drmm_config import config
import torch

        
def get_model_size(model):
    return sum([ p.size(0) if len(p.size()) == 1 else p.size(0)*p.size(1) for p in model.parameters()])

def validate_tv(drmm, validationloader, criterion, histograms, embeddings):

    validation_losses = []
    
    for i, data in enumerate(validationloader, 0):
    
        queries, docs_h, docs_l = data
        
        queries_vectors = [embeddings[q] for q in queries]
        histograms_h = [histograms[qid][doc] for qid, doc in zip(queries, docs_h)]
        histograms_l = [histograms[qid][doc] for qid, doc in zip(queries, docs_l)]

        histograms_h, histograms_l, queries_tvs = process_minibatch(histograms_h, histograms_l, queries_vectors)
        
        score_h = drmm(histograms_h,queries_tvs)
        score_l = drmm(histograms_l,queries_tvs)
        
        loss = criterion(score_h,score_l)
        validation_losses.append(loss.data)

    return np.mean(validation_losses)

def train_tv(train_file, validation_file, histogram_file, save_dir, experiment_name):
    
    logging.basicConfig(filename=experiment_name, level=logging.INFO)
    logger = logging.getLogger(experiment_name)
    
    drmm = DRMM_TV()
    drmm = torch.nn.DataParallel(drmm, device_ids=[0, 1, 2])
    criterion = HingeLoss().cuda()
    optimizer = torch.optim.Adam(drmm.parameters(),lr = 0.0001)
    logger.info("Starting {} experiment with {} parameters".format(experiment_name, get_model_size(drmm)))
    logger.info("Training from {} and validation from {}.".format(train_file,validation_file))
    logger.info("Loading histograms from {}.".format(histogram_file))
    histograms = load_histograms(histogram_file)
   
    logger.info("Loading word embeddings from {}.".format(config["queries_tv"]))
    embeddings = {line.split(" ",1)[0]: json.loads(line.strip().split(" ",1)[1]) for line in open(config["queries_tv"])}
    
    logger.info("Loading training pairs generator.")
    train_generator = PairsGenerator(config["pairs_file"],train_file)
    logger.info("Loading validation pairs generator.")
    validation_generator = PairsGenerator(config["pairs_file"],validation_file)

    trainloader = torch.utils.data.DataLoader(train_generator, batch_size=30, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_generator, batch_size=20)

    for epoch in range(30):
        
        start_time = time.time()
        train_loss = []
        c = 0
        sum_total_time = 0.0
        sum_time_proc_input = 0.0
        sum_time_variable_data = 0.0
        sum_time_training = 0.0
        sum_time_proc_hist = 0.0
        for i, data in enumerate(trainloader, 0):
            total_time = time.time()
            queries, docs_h, docs_l = data
            time_proc_input = time.time()            
            histograms_h = [histograms[qid][doc] for qid, doc in zip(queries, docs_h)]
            histograms_l = [histograms[qid][doc] for qid, doc in zip(queries, docs_l)]
            
            queries_vectors = [ embeddings[qid] for qid, doc in zip(queries, docs_l)]
            time_data = time.time()
            histograms_h, histograms_l, queries_tvs, t = process_minibatch(histograms_h, histograms_l, queries_vectors)
            sum_time_proc_hist += t
            sum_time_variable_data += (time.time() - time_data)
            sum_time_proc_input += (time.time() - time_proc_input)
            time_proc_train = time.time() 
            score_h = drmm(histograms_h,queries_tvs)
            score_l = drmm(histograms_l,queries_tvs)
            
            optimizer.zero_grad()
            loss = criterion(score_h,score_l)
            train_loss.append(loss.data)
            loss.backward()
            optimizer.step()	
            sum_time_training += (time.time() - time_proc_train)

            sum_total_time += (time.time() - total_time)
            c += 20
            if c % 10000 == 0:
                print('{:.3f} {:.3f}'.format((time.time() - start_time)/60, 100*c/train_generator.len))
                print('{:.2f} {:.2f}'.format(100*sum_time_proc_input/sum_total_time, 100*sum_time_training/sum_total_time))
                print('{:.2f} {:.2f}'.format(100*sum_time_variable_data/sum_total_time, 100*sum_time_proc_hist/sum_total_time))
                print('TrainingLoss: {}'.format(np.mean(train_loss)))
        time_training = (time.time() - start_time)
        #validation_loss = validate_tv(drmm, validationloader, criterion, histograms, embeddings)
        #logger.info('Epoch : {}\tTrainingLoss: {}\tValidationLoss: {}\tTime: {}'.format(epoch, np.mean(train_loss).numpy()[0],
        #        validation_loss.numpy()[0],time_training))
        
        torch.save(
            drmm.state_dict(),
            open(os.path.join(
                save_dir,
                experiment_name + '_epoch_%d' % (epoch) + '.model'), 'wb'
            )
        )


def main():
    # train_file, validation_file, histogram_file,  save_dir, experiment_name
    train_tv(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4], sys.argv[5])
    
main()
