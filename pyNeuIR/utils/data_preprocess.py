import torch
from torch.autograd import Variable
import torch.nn as nn
import operator
import numpy as np


def construct_vocab(lines, vocab_size):
    """Construct a vocabulary from tokenized lines."""
    vocab = {}
    for line in lines:
        for word in line:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    # Discard start, end, pad and unk tokens if already present
    if '<pad>' in vocab:
        del vocab['<pad>']
    if '<unk>' in vocab:
        del vocab['<unk>']

    word2id = {
        '<pad>': 0,
        '<unk>': 1,
    }

    id2word = {
        1: '<pad>',
        3: '<unk>',
    }

    sorted_word2id = sorted(
        vocab.items(),
        key=operator.itemgetter(1),
        reverse=True
    )

    sorted_words = [x[0] for x in sorted_word2id[:vocab_size]]

    for ind, word in enumerate(sorted_words):
        word2id[word] = ind + 2

    for ind, word in enumerate(sorted_words):
        id2word[ind + 2] = word

    return word2id, id2word

def load_word_embeddings(vocab_size, emb_dim):
    embeddings = nn.Embedding(
        vocab_size,
        emb_dim
    )
    return embeddings

