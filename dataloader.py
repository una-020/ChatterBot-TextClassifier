from torch.utils.data import Dataset
from utils import *

import torch
import numpy as np


class TextDataset(Dataset):
    def __init__(self, X, Y, wv_model):
        self.X = X
        self.Y = Y
        self.wv_model = wv_model
        self.embed_dim = len(wv_model["king"])
        self.max_len = get_max_len(X)

    def __getitem__(self, index):
        sentence = self.X[index]
        
        wv_model = self.wv_model
        max_len = self.max_len
        
        sent_embed = np.array([]).reshape(self.embed_dim, 0)

        sent_embed = []
        for word in sentence.split():
            try:
                word_embed = wv_model.word_vec(word)
            except KeyError:
                word_embed = self.process_unk(wv_model, word)

            sent_embed.append(word_embed)

        for i in range(self.max_len - len(sent_embed)):
            sent_embed.append(np.zeros(self.embed_dim))
        sent_embed = np.array(sent_embed)
        sent_embed = torch.from_numpy(sent_embed)
        
        return (sent_embed, self.Y[index])

    def __len__(self):
        return len(self.X)
    
    def process_unk(self, wv_model, word):
        word_embed = np.zeros((self.embed_dim))
        for i in range(len(word)):
            try:
                word_embed += ((i + 1) * wv_model.word_vec(word[i]))
            except KeyError:
                continue

        n = len(word)
        word_embed /= (n * (n + 1)) / 2
        return word_embed
