from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
from sklearn import preprocessing
from dataloader import *
from utils import *

import torch
import numpy as np
import torch.nn as nn


class Model:
    def train_model(self, X, Y, **kwargs):
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

    def eval_model(self, y_true, y_pred, average=None):
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

    def train_model(self, X, Y, **kwargs):
        self.cls.fit(X, Y)

    def predict(self, X, **kwargs):
        return self.cls.predict(X)

    def get_X(self, sentence_list, **kwargs):
        fit = kwargs.get("fit", True)
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
        self.wv_model = kwargs.get("wv_model")
        dropout = kwargs.get("dropout", 0.5)
        num_layers = kwargs.get("num_layers", 1)
        if num_layers == 1:
            dropout = 0

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

    def train_model(self, X, Y, **kwargs):
        dataset = TextDataset(X, Y, self.wv_model)
        text_loader = get_dataloader(dataset,
                                     batch_size=kwargs.get("batch_size", 128),
                                     num_workers=kwargs.get("num_workers", 4)
                                    )
        
        for i, (x_batch, y_batch) in enumerate(text_loader):
            print(x_batch.shape, y_batch.shape, i)

    def predict(self, X, **kwargs):
        pass

    def get_X(self, sentence_list, **kwargs):
        return sentence_list
    
    def save_corpus(self, corpus_name, X, Y):
        if not os.path.exists("pkl_files/"):
            os.mkdir("pkl_files")
        n_file = "pkl_files/nn_" + corpus_name + ".pkl"
        pkl.dump([X, Y, self.labeler], open(n_file, "wb"))

    def load_corpus(self, corpus_name):
        bundle = pkl.load(open("pkl_files/nn_" + corpus_name + ".pkl", "rb"))
        X, Y, self.labeler = bundle
        return X, Y
