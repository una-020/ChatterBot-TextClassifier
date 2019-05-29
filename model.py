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
import torch.optim as optim


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
        self.hidden_dim = kwargs.get("hidden_dim", 100)
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
                                     num_workers=kwargs.get("num_workers", 4),
                                     shuffle=True
                                    )
        
        optimizer = optim.Adam(self.parameters(), lr=0.1)
        loss_function = nn.CrossEntropyLoss()
        
        for epoch in range(1, kwargs.get("num_epochs", 100) + 1):
            print("Epoch " + str(epoch))
            correct_train = 0      
            self.train()
            for i, (data, label) in enumerate(text_loader):
                data = data.type(torch.FloatTensor)
                data = data.cuda()
                label = label.cuda()
                
                optimizer.zero_grad()
                output = self.forward(data)
                loss = loss_function(output, label)
                loss.backward()
                optimizer.step()
                _, idx = torch.max(output, dim=1)
                correct_train += (idx == label).sum().item()
                print("\rBatch %d: %f %d" % (i, loss.item(), correct_train), end="")
                
            print("\nTraining Accuracy:", correct_train / len(text_loader))

            

    def predict(self, X, **kwargs):
        p = self.forward(X)
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
