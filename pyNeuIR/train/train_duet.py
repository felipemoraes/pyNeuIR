import sys
import os
import logging
import time
import numpy as np

sys.path.append('../../pyNeuIR/')

from pyNeuIR.utils.preprocess import load_features_local, load_features_distrib_query, load_features_distrib_doc
from pyNeuIR.utils.pairs_generator import PairsGenerator
from pyNeuIR.models.duet import Duet
from pyNeuIR.configs.duet_config import config
import torch
from torch.autograd import Variable
torch.manual_seed(222)

use_cuda = torch.cuda.device_count() > 0
        
if use_cuda:
    torch.cuda.manual_seed(222)

def get_model_size(model):
    return sum([ p.size(0) if len(p.size()) == 1 else p.size(0)*p.size(1) for p in model.parameters()])

def train(trainloader, validationloader, features_local, 
    features_distrib_query, features_distrib_doc,  save_dir, experiment_name):
    
    global logger
    
    duet = Duet(10,1000,2000)
    target = Variable((torch.ones(8)-1).type(torch.LongTensor), requires_grad=False)
    if use_cuda:
        duet = duet.cuda()
        target = target.cuda()
    
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(duet.parameters(),lr = 0.01)

    logger.info("Start training {} experiment with {} parameters".format(experiment_name, get_model_size(duet)))
    
    for epoch in range(10):
        train_loss = []
        time_start = time.time()
        for i, data in enumerate(trainloader, 0):

            queries, docs_pairs = data
            
            queries_features_distrib = torch.stack([features_distrib_query[query] for query in queries])
            queries_features_distrib = Variable(queries_features_distrib.unsqueeze(1))
            if use_cuda:
                queries_features_distrib = queries_features_distrib.cuda()
            all_scores = []
            
            for docs_pair in docs_pairs:
                queryies_docs_features_local = torch.stack([features_local[qid][doc] for qid, doc in zip(queries, docs_pair)])
                docs_features_distrib = torch.stack([features_local[qid][doc] for qid, doc in zip(queries, docs_pair)])

                queryies_docs_features_local = Variable(queryies_docs_features_local.unsqueeze(1))
                docs_features_distrib = Variable(docs_features_distrib.unsqueeze(1))

                if use_cuda:
                    queryies_docs_features_local = queryies_docs_features_local.cuda()
                    docs_features_distrib = docs_features_distrib.cuda()
                scores = duet(queryies_docs_features_local, queries_features_distrib, docs_features_distrib) 
                all_scores.append(duet(scores))
            
            
            optimizer.zero_grad()
            all_scores = torch.stack(all_scores).permute(1,0)
            loss = criterion(all_scores,target)
            train_loss.append(loss.data)
            loss.backward()
            optimizer.step()	
       
        time_training = time.time() - time_start
        #validation_loss = validate(drmm, criterion, validationloader, histograms, embeddings)
        logger.info('Epoch : {}\tTrainingLoss: {}\tValidationLoss: {}\tTime: {}'.format(epoch, np.mean(train_loss).cpu().numpy()[0],
                validation_loss.cpu().numpy()[0],time_training))
        
        torch.save(
            duet.state_dict(),
            open(os.path.join(
                save_dir,
                experiment_name + '_epoch_%d' % (epoch) + '.model'), 'wb'
            )
        )

def validate(duet, criterion, validationloader, features_local, features_distrib_query, features_distrib_doc):

    validation_losses = []

    for i, data in enumerate(trainloader, 0):

        queries, docs_pairs = data
        
        queries_features_distrib = torch.stack([features_distrib_query[query] for query in queries])
        queries_features_distrib = Variable(queries_features_distrib.unsqueeze(1))
        if use_cuda:
            queries_features_distrib = queries_features_distrib.cuda()
        all_scores = []
        
        for docs_pair in docs_pairs:
            queryies_docs_features_local = torch.stack([features_local[qid][doc] for qid, doc in zip(queries, docs_pair)])
            docs_features_distrib = torch.stack([features_local[qid][doc] for qid, doc in zip(queries, docs_pair)])

            queryies_docs_features_local = Variable(queryies_docs_features_local.unsqueeze(1))
            docs_features_distrib = Variable(docs_features_distrib.unsqueeze(1))

            if use_cuda:
                queryies_docs_features_local = queryies_docs_features_local.cuda()
                docs_features_distrib = docs_features_distrib.cuda()
            scores = duet(queryies_docs_features_local, queries_features_distrib, docs_features_distrib) 
            all_scores.append(duet(scores))
        
    
        all_scores = torch.stack(all_scores).permute(1,0)
        loss = criterion(all_scores,target)
        validation_losses.append(loss.data)

    return np.mean(validation_losses)

def main():

    train_file = sys.argv[1]
    validation_file = sys.argv[2]
    save_dir = sys.argv[3]
    experiment_name = sys.argv[4]
    #filename=experiment_name + ".log"
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    global logger
    logger = logging.getLogger(experiment_name)
    
    logger.info("Training from {} and validation from {}.".format(train_file,validation_file))
    
    logger.info("Loading features local.")

    features_local = load_features_local(config["features_local"], config["n_d"],  config["n_q"], train_file, validation_file)
    logger.info("Loading features distrib query.")
    
    features_distrib_query = load_features_distrib_query(config["features_distrib_query_file"], config["m_d"],  config["n_q"], train_file, validation_file)

    logger.info("Loading features distrib doc.")

    features_distrib_doc = load_features_distrib_doc(config["features_distrib_doc_file"], config["m_d"],  config["n_d"], train_file, validation_file)
   

    logger.info("Loading training pairs generator.")
    train_generator = PairsGenerator(config["pairs_file"],train_file)
    
    logger.info("Loading validation pairs generator.")
    validation_generator = PairsGenerator(config["pairs_file"],validation_file)

    trainloader = torch.utils.data.DataLoader(train_generator, batch_size=8, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_generator, batch_size=8)
    
    train(trainloader, validationloader, features_local, features_distrib_query, features_distrib_doc,  save_dir, experiment_name)
    
main()
