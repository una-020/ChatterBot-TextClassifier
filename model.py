from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
from sklearn import preprocessing
from utils import *

import torch
import numpy as np
import torch.nn as nn


class Model:
    def train(self, X, Y, **kwargs):
        pass

    def predict(self, X, **kwargs):
        pass

    def get_X(self, sentence_list, **kwargs):
        pass

    def get_Y(self, label_list, fit=True):
        if fit:
            self.labeler = preprocessing.LabelEncoder()
            Y = self.labeler.fit_transform(label_list)
        else:
            Y = self.labeler.transform(label_list)
        return Y

    def eval(self, y_true, y_pred, average=None):
        assert len(y_true) == len(y_pred)
        precision, recall, fscore, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=average
        )
        accuracy = accuracy_score(y_true, y_pred)
        return precision, recall, fscore, accuracy
    
    def save_corpus(self, corpus_name, X, Y):
        pass
    
    def load_corpus(self, corpus_name):
        pass


class Logistic(Model):
    def __init__(self, C):
        self.cls = LogisticRegression(
            random_state=0,
            C=C,
            solver='lbfgs',
            max_iter=10000,
            n_jobs=-1,
            class_weight='balanced',
            multi_class='auto'
        )

    def train(self, X, Y, **kwargs):
        self.cls.fit(X, Y)

    def predict(self, X, **kwargs):
        return self.cls.predict(X)

    def get_X(self, sentence_list, **kwargs):
        fit = kwargs.get("fit", True)
        sentence_list = preprocess_periods(sentence_list)
        if fit:
            self.vect = TfidfVectorizer(
                tokenizer=word_tokenize,
            )
            X = self.vect.fit_transform(sentence_list)
        else:
            X = self.vect.transform(sentence_list)
        return X

    def save_corpus(self, corpus_name, X, Y):
        if not os.path.exists("pkl_files/"):
            os.mkdir("pkl_files")
        n_file = "pkl_files/lr_" + corpus_name + ".pkl"
        pkl.dump([X, Y, self.vect, self.labeler], open(n_file, "wb"))

    def load_corpus(self, corpus_name):
        bundle = pkl.load(open("pkl_files/lr_" + corpus_name + ".pkl", "rb"))
        X, Y, self.vect, self.labeler = bundle
        return X, Y


class Lstm(Model, nn.Module):
    def __init__(self, **kwargs):
        super(Lstm, self).__init__()

        self.embed_dim = kwargs.get("embed_dim")
        self.hidden_dim = kwargs.get("hidden_dim", 600)
        self.num_classes = kwargs.get("num_classes")
        dropout = kwargs.get("dropout", 0.5)
        num_layers = kwargs.get("num_layers", 1)

        self.lstm = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout
        )

        self.decoder = nn.Linear(
            in_features=self.hidden_dim,
            out_features=self.num_classes
        )

    def forward(self, X):
        o, _ = self.lstm(X)
        o = o[:, -1, :]
        o = o.contiguous().view(-1, self.hidden_dim)
        o = self.decoder(o)
        return o

    def train(self, X, Y, **kwargs):
        class WVECDataset(Dataset):
            def __init__(self, X, Y):
                self.X = X
                self.Y = Y
                self.max_len = get_max_len(X)

            def __getitem__(self, index_wvec):
                feature = self.X[index]
                return (self.X[index, :], self.Y[index])

    def __len__(self):
        return self.X.shape[0]

            
        batch_size = kwargs.get("batch_size", 128)
        num_workers = kwargs.get("num_workers", 4)

        text_loader = get_dataloader(X, Y, batch_size=batch_size,
                                     num_workers=num_workers)

        for i, (x_batch, y_batch) in enumerate(text_loader):
            # use x_batch and y_batch to train the network
            # x_batch : batch_size x em_size x max_sen_len, y_batch: batch_size
            pass

    def predict(self, X, **kwargs):
        pass

    def get_X(self, sentence_list, **kwargs):
        wv_model = kwargs.get("wv_model")
        sentence_list = preprocess_sentences(sentence_list)
        max_len = kwargs.get("max_len")

        X = []
        for i in range(len(sentence_list)):
            sentence = sentence_list[i]
            sent_embed = np.array([]).reshape(self.embed_dim, 0)

            for word in sentence.split():
                try:
                    word_embed = wv_model.word_vec(word)
                except KeyError:
                    word_embed = self.process_unk(wv_model, word)

                word_embed = word_embed.reshape(-1, 1)
                sent_embed = np.hstack((sent_embed, word_embed))

            sent_embed = np.pad(sent_embed,
                                ((0, 0), (0, max_len - sent_embed.shape[-1])),
                                'constant')

            X.append(sent_embed)
        X = np.array(X)
        X = X.permute(0, 2, 1)
        return torch.from_numpy(X)

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
