from torch.utils.data import Dataset
from random import randint

class MockPairGenerator(Dataset):

    def __init__(self, vocab, query_len, doc_len, samples):

        self.vocab = vocab
        self.v_len = len(self.vocab)
        self.len = samples
        self.query_len = query_len
        self.doc_len = doc_len

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        q_len = randint(1,self.query_len)
        
        dh_len = randint(1,self.doc_len)
        dl_len = randint(1,self.doc_len)
        query = [ self.vocab[randint(0,self.v_len-1)] for _ in range(q_len)]
        dh = [ self.vocab[randint(0,self.v_len-1)] for _ in range(dh_len)]
        dl = [ self.vocab[randint(0,self.v_len-1)] for _ in range(dl_len)]
        
        return " ".join(query), " ".join(dh), " ".join(dl)
        

        
        