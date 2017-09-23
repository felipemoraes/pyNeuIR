from torch.utils.data import Dataset
import random

class PairsGenerator(Dataset):

    def __init__(self, pairs_file, train_file):
        
        topics = set([line.strip().split()[0] for line in open(train_file)])
        self.pairs = []

        for line in open(pairs_file):
            tid, high_d, low_d = line.strip().split()
            if tid in topics:
                self.pairs.append((tid, high_d, low_d))
        
        self.len = len(self.pairs)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.pairs[idx]

        
        