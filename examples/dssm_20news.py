import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
import string
import random
import os
import sys
import numpy as np

sys.path.append('../pyNeuIR/')

from pyNeuIR.models.dssm import DSSM, LogLoss

def letter_ngrams(word, n=3):
	ngrams = []
	for i in range(0,len(word)-3+1):
		ngrams.append(word[i:i+3])
	return ngrams

def letter_ngram_tokenizer(s):
	letter_ngram = []
	for word in word_tokenize(s):
		word = word.lower()
		word = word.translate(str.maketrans('', '', string.punctuation))
		if len(word):
			word = "#" + word + "#"
			letter_ngram.extend(letter_ngrams(word))

	return letter_ngram

class Fetcher20newsgroupsDataset(Dataset):
	"""20newsgroups dataset."""

	def group_by_category(self,dataset):
		group_categories = [[],[],[],[]]
		for i, instance in enumerate(dataset.target):
			group_categories[instance].append(i)
		return group_categories

	def __init__(self,subset):
		self.subset = subset
		categories = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
		trainset = fetch_20newsgroups(subset='train',categories=categories)
		vectorizer = CountVectorizer(tokenizer=letter_ngram_tokenizer)
		self.train_instances = []
		train_cat = self.group_by_category(trainset)
		for c in range(4):
			all_cat_minus_c = []
			for i in range(4):
				if i != c:
					all_cat_minus_c.extend(train_cat[i])
			
			for i, v  in enumerate(train_cat[c]):
				pos_doc = random.sample(train_cat[c][:i] + train_cat[c][i+1:],1)[0]
				neg_docs = random.sample(all_cat_minus_c,4)
				self.train_instances.append([i, pos_doc] + neg_docs)
		
		self.train_vectors = vectorizer.fit_transform(trainset.data)
		self.len = len(self.train_instances)
		self.vocab_size = self.train_vectors.shape[1]
		print(self.len)
	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		idxs = self.train_instances[idx] 
		return [self.train_vectors[i].toarray()[0].astype(np.float32, copy=False) for i in idxs]

	def get_vector(self, x):
		return self.vectorizer.transform(x)

trainset = Fetcher20newsgroupsDataset("train")

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)



dssm = DSSM(0.005, [trainset.vocab_size,300,300,128])
criterion = LogLoss()
optimizer = torch.optim.Adam(dssm.parameters(),lr = 0.0005)

for epoch in range(100):  # loop over the dataset multiple times

	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# get the inputs
		query, doc1, doc2, doc3, doc4, doc5 = data

		# wrap them in Variable
		query, doc1, doc2, doc3, doc4 = Variable(query), Variable(doc1), Variable(doc2), Variable(doc3), Variable(doc4)
		doc5 = Variable(doc5)
		output = dssm(query, doc1, doc2, doc3, doc4,doc5)
		optimizer.zero_grad()
		loss_log = criterion(output)
		loss_log.backward()
		optimizer.step()
		
		if i %10 == 0 :
			print("Epoch number {}\n Current loss {}\n".format(epoch,loss_log.data[0]))

		