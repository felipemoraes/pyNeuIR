import sys
import os
import logging
import time
import json
import numpy as np

sys.path.append('../../pyNeuIR/')

from pyNeuIR.utils.preprocess import pad_sequences, load_embeddings, load_histograms
from pyNeuIR.utils.pairs_generator import PairsGenerator
from pyNeuIR.models.drmm import DRMM_TV, HingeLoss
from pyNeuIR.configs.drmm_config import config
import torch

        
def get_model_size(model):
    return sum([ p.size(0) if len(p.size()) == 1 else p.size(0)*p.size(1) for p in model.parameters()])

def train_tv(trainloader, validationloader, histograms, embeddings, save_dir, experiment_name
    
    global logger

    drmm = DRMM_TV().cuda()
    criterion = HingeLoss().cuda()
    optimizer = torch.optim.Adam(drmm.parameters(),lr = 0.0001)

logger.info("Start training {} experiment with {} parameters".format(experiment_name, get_model_size(drmm)))

    for epoch in range(30):
        train_loss = []
        for i, data in enumerate(trainloader, 0):

            queries, docs_h, docs_l = data
                     
            histograms_h = [histograms[qid][doc] for qid, doc in zip(queries, docs_h)]
            histograms_l = [histograms[qid][doc] for qid, doc in zip(queries, docs_l)]
            queries_tvs = [embeddings[qid] for qid in queries]
            
            histograms_h = Variable(pad_sequences(histograms_h))
            histograms_l = Variable(pad_sequences(histograms_l))
            queries_tvs = Variable(pad_sequences(queries_tvs))

            if use_cuda:
                histograms_h = histograms_h.cuda()
                histograms_l = histograms_l.cuda()
                queries_tvs = queries_tvs.cuda()

            score_h = drmm(histograms_h,queries_tvs)
            score_l = drmm(histograms_l,queries_tvs)
            
            optimizer.zero_grad()
            loss = criterion(score_h,score_l)
            train_loss.append(loss.data)
            loss.backward()
            optimizer.step()	
        
        validation_loss = validate_tv(drmm, validationloader, criterion, histograms, embeddings)
        #logger.info('Epoch : {}\tTrainingLoss: {}\tValidationLoss: {}\tTime: {}'.format(epoch, np.mean(train_loss).numpy()[0],
        #        validation_loss.numpy()[0],time_training))
        
        torch.save(
            drmm.state_dict(),
            open(os.path.join(
                save_dir,
                experiment_name + '_epoch_%d' % (epoch) + '.model'), 'wb'
            )
        )

def validate_tv(drmm, validationloader, criterion, histograms, embeddings):

    validation_losses = []
    
    for i, data in enumerate(validationloader, 0):
    
        queries, docs_h, docs_l = data
                     
        histograms_h = [histograms[qid][doc] for qid, doc in zip(queries, docs_h)]
        histograms_l = [histograms[qid][doc] for qid, doc in zip(queries, docs_l)]
        queries_tvs = [embeddings[qid] for qid in queries]
        
        histograms_h = Variable(pad_sequences(histograms_h))
        histograms_l = Variable(pad_sequences(histograms_l))
        queries_tvs = Variable(pad_sequences(queries_tvs))

        if use_cuda:
            histograms_h = histograms_h.cuda()
            histograms_l = histograms_l.cuda()
            queries_tvs = queries_tvs.cuda()

        score_h = drmm(histograms_h,queries_tvs)
        score_l = drmm(histograms_l,queries_tvs)
        
        loss = criterion(score_h,score_l)
        validation_losses.append(loss.data)

    return np.mean(validation_losses)

def main():

    train_file = sys.argv[1]
    validation_file = sys.argv[2]
    histogram_file = sys.argv[3]
    save_dir = sys.argv[4]
    experiment_name = sys.argv[5]
    type_term_gating = sys.argv[6]

    logging.basicConfig(filename=experiment_name + ".log", level=logging.INFO)
    global logger
    logger = logging.getLogger(experiment_name)
    
    logger.info("Training from {} and validation from {}.".format(train_file,validation_file))
    logger.info("Loading histograms from {}.".format(histogram_file))
    
    histograms = load_histograms(histogram_file)
   
    logger.info("Loading word embeddings from {}.".format(config["queries_tv"]))
    embeddings = load_embeddings(config["queries_tv"])
    
    logger.info("Loading training pairs generator.")
    train_generator = PairsGenerator(config["pairs_file"],train_file)
    
    logger.info("Loading validation pairs generator.")
    validation_generator = PairsGenerator(config["pairs_file"],validation_file)

    trainloader = torch.utils.data.DataLoader(train_generator, batch_size=30, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_generator, batch_size=len(validation_generator))
    
    if type_term_gating == "tv":
        logger.info("Loading word embeddings from {}.".format(config["queries_tv"]))
        embeddings = load_embeddings(config["queries_tv"])
        logger.info("Start training experiment with {} ".format(experiment_name, get_model_size(drmm)))
        train_tv(trainloader, validationloader, histograms, embeddings, save_dir, experiment_name)
    elif type_term_gating == "idf":
        #TODO
        logger.info("Loading word embeddings from {}.".format(config["queries_tv"]))
        embeddings = load_embeddings(config["queries_tv"])
        logger.info("Start training experiment with {} ".format(experiment_name, get_model_size(drmm)))
        train_tv(trainloader, validationloader, histograms, embeddings, save_dir, experiment_name)
    
main()
