import sys
import os
import logging
import time
import numpy as np

sys.path.append('../../pyNeuIR/')

from pyNeuIR.utils.preprocess import pad_sequences, load_embeddings, load_histograms
from pyNeuIR.utils.pairs_generator import PairsGenerator
from pyNeuIR.models.drmm import DRMM, HingeLoss
from pyNeuIR.configs.drmm_config import config
import torch
from torch.autograd import Variable
torch.manual_seed(222)

use_cuda = torch.cuda.device_count() > 0
        
if use_cuda:
    torch.cuda.manual_seed(222)

def get_model_size(model):
    return sum([ p.size(0) if len(p.size()) == 1 else p.size(0)*p.size(1) for p in model.parameters()])

def train(trainloader, validationloader, histograms, embeddings, save_dir, experiment_name):
    
    global logger
    
    drmm = DRMM(300)
    if use_cuda:
        drmm = drmm.cuda()
    criterion = HingeLoss()

    optimizer = torch.optim.Adagrad(drmm.parameters(),lr = 0.001)
    
    logger.info("Start training {} experiment with {} parameters".format(experiment_name, get_model_size(drmm)))

    for epoch in range(100):
        train_loss = []
        time_start = time.time()
        for i, data in enumerate(trainloader, 0):

            queries, docs_h, docs_l = data
                     
            histograms_h = torch.stack([histograms[qid][doc] for qid, doc in zip(queries, docs_h)])
            histograms_l = torch.stack([histograms[qid][doc] for qid, doc in zip(queries, docs_l)])
            queries_tvs = torch.stack([embeddings[qid] for qid in queries])
            
            histograms_h = Variable(histograms_h)
            histograms_l = Variable(histograms_l)
            queries_tvs = Variable(queries_tvs)


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
        time_training = time.time() - time_start
        validation_loss = validate(drmm, criterion, validationloader, histograms, embeddings)
        logger.info('Epoch : {}\tTrainingLoss: {}\tValidationLoss: {}\tTime: {}'.format(epoch, np.mean(train_loss).cpu().numpy()[0],
                validation_loss.cpu().numpy()[0],time_training))
        
        torch.save(
            drmm.state_dict(),
            open(os.path.join(
                save_dir,
                experiment_name + '_epoch_%d' % (epoch) + '.model'), 'wb'
            )
        )

def validate(drmm, criterion, validationloader, histograms, embeddings):

    validation_losses = []

    for i, data in enumerate(validationloader, 0):
    
        queries, docs_h, docs_l = data
                     
        histograms_h = torch.stack([histograms[qid][doc] for qid, doc in zip(queries, docs_h)])
        histograms_l = torch.stack([histograms[qid][doc] for qid, doc in zip(queries, docs_l)])
        queries_tvs = torch.stack([embeddings[qid] for qid in queries])
        histograms_h = Variable(histograms_h, requires_grad=False)
        histograms_l = Variable(histograms_l, requires_grad=False)
        queries_tvs = Variable(queries_tvs, requires_grad=False)

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
    logging.basicConfig(filename=experiment_name + ".log",level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    global logger
    logger = logging.getLogger(experiment_name)
    
    logger.info("Training from {} and validation from {}.".format(train_file,validation_file))
    logger.info("Loading histograms from {}.".format(histogram_file))
    
    histograms = load_histograms(histogram_file,5)
   
    logger.info("Loading word embeddings from {}.".format(config["queries_tv"]))
    embeddings = load_embeddings(config["queries_tv"],5)

    logger.info("Loading training pairs generator.")
    train_generator = PairsGenerator(config["pairs_file"],train_file)
    
    logger.info("Loading validation pairs generator.")
    validation_generator = PairsGenerator(config["pairs_file"],validation_file)

    trainloader = torch.utils.data.DataLoader(train_generator, batch_size=20, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_generator, batch_size=len(validation_generator))
    
    train(trainloader, validationloader, histograms, embeddings, save_dir, experiment_name)
    
main()
