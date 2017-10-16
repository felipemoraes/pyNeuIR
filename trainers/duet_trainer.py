import sys
import os
import time
import numpy as np
import argparse

sys.path.append('.')

from models.duet import Duet
from configs.duet_config import config as c
from datasets.duet_dataset import DuetDataset
import torch
from torch.autograd import Variable
import torch.nn as nn
torch.manual_seed(222)

use_cuda = torch.cuda.device_count() > 0
        
if use_cuda:
    torch.cuda.manual_seed(222)

def get_model_size(model):
    return sum([ p.size(0) if len(p.size()) == 1 else p.size(0)*p.size(1) for p in model.parameters()])

def to_cuda(variable):
    if use_cuda:
        return variable.cuda()
    return variable

def trainer(dataloader, save_dir, experiment_name, num_docs, model_type="duet"):
    
    duet = Duet(c["n_q"],c["n_d"], c["m_d"], model_type)
    if use_cuda:
        duet = duet.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(duet.parameters(),lr = 0.01)

    print("Started trainer {} experiment with {} parameters".format(experiment_name, get_model_size(duet)))
    
    for epoch in range(c['epoch']):
        train_loss = []
        for i, data in enumerate(dataloader, 0):
            time_start = time.time()

            # Initialize variables
            features_local = None
            features_distrib_query = None
            features_distrib_doc = None

            # Load variables according to the model type
            if model_type == "local" or model_type == "duet":
                features_local = data["features_local"]
                features_local = [to_cuda(Variable(features_local[:,i,:,:]).unsqueeze(1)) for i in range(num_docs)]

            if model_type == "distrib" or model_type == "duet":
                features_distrib_query = data["features_distrib_query"]
                features_distrib_doc = data["features_distrib_doc"]
                features_distrib_query = to_cuda(Variable(features_distrib_query).unsqueeze(1))
                features_distrib_doc = [to_cuda(Variable(features_distrib_doc[:,i,:,:]).unsqueeze(1)) for i in range(num_docs)]
            
            labels = Variable(data["labels"].type(torch.LongTensor), requires_grad=False)

            output = duet(features_local, features_distrib_query, features_distrib_doc, num_docs)

            optimizer.zero_grad()
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.data)

            time_training = time.time() - time_start
            print('Epoch : {} Minibatch: {} \tTraining Loss: {}\tTraining Time: {}'.format(epoch, i, np.mean(train_loss).cpu().numpy()[0],time_training))
        
        torch.save(
             duet.state_dict(),
             open(os.path.join(
                 save_dir,
                 experiment_name + '_epoch_%d' % (epoch) + '.model'), 'wb'
             )
        )

    
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', help="Dataset folder")

    parser.add_argument('-train', help="Train queries")

    parser.add_argument('-o', help="Output folder")

    parser.add_argument('-name', help="Prefix experiment name")

    parser.add_argument('-type', help="Type of network", choices=['duet', 'local', 'distrib'], default="duet")

    args = parser.parse_args()

    print("Loading dataset.")

    dataset = DuetDataset(args.dataset, args.train, args.type)

    if not os.path.exists(args.o):
        os.makedirs(args.o)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    trainer(dataloader, args.o, args.name, 5, args.type)
    
main()
