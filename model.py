from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
from sklearn import preprocessing
from dataloader import *

import torch
import numpy as np
import pickle as pkl
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
    
    def save_model(self, corpus_name):
        pass
    
    def load_model(self, corpus_name):
        pass


class Logistic(Model):
    def __init__(self, C):
        self.C = C
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
        n_file = "pkl_files/lr_" + corpus_name + ".pkl"
        bundle = pkl.load(open(n_file, "rb"))
        X, Y, self.vect, self.labeler = bundle
        return X, Y
    
    def save_model(self, corpus_name):
        if not os.path.exists("pkl_files/"):
            os.mkdir("pkl_files")
        n_file = "pkl_files/lr_" + corpus_name + "_model.pkl"
        pkl.dump(self.cls, open(n_file, "wb"))
        params = {"C": self.C}
        p_file = "pkl_files/lr_" + corpus_name + "_params.pkl"
        pkl.dump(params, open(p_file, "wb"))
    
    def load_model(self, corpus_name):
        n_file = "pkl_files/lr_" + corpus_name + "_model.pkl"
        self.cls = pkl.load(open(n_file, "rb"))


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
        self.dropout = dropout
        self.num_layers = num_layers

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
        
        self.initialise_parameters()
            

    def forward(self, X):
        o, _ = self.lstm(X)
        o = o[:, -1, :]
        o = o.contiguous().view(-1, self.hidden_dim)
        o = self.decoder(o)
        return o
    
    def initialise_parameters(self):
        nn.init.xavier_uniform_(self.decoder.weight)

    def train_model(self, X, Y, **kwargs):
        is_cuda = next(self.parameters()).is_cuda
        device = 'cuda' if is_cuda else 'cpu'

        dataset = TextDataset(X, Y, self.wv_model)
        text_loader = get_dataloader(
            dataset,
            batch_size=kwargs.get("batch_size", 128),
            num_workers=kwargs.get("num_workers", 4),
            shuffle=True
        )

        optimizer = optim.Adam(self.parameters(), lr=0.001)
        loss_function = nn.CrossEntropyLoss()
        for epoch in range(1, kwargs.get("epoch", 100) + 1):
            print("Epoch " + str(epoch))
            correct_train = 0      
            self.train()
            for i, (data, label) in enumerate(text_loader):
                data = data.type(torch.FloatTensor)
                data = data.to(device)
                label = label.to(device)
                optimizer.zero_grad()
                output = self.forward(data)
                loss = loss_function(output, label)
                loss.backward()
                optimizer.step()
                _, idx = torch.max(output, dim=1)
                correct_train += (idx == label).sum().item()
                print("\rBatch %d: %f" % (i, loss.item()), end="")
            print("\nTraining Accuracy:", correct_train / len(dataset))

    def predict(self, X, **kwargs):
        device = 'cuda' if is_cuda else 'cpu'
        label_dummy = [0] * len(X)
        wv_model = kwargs.get("wv_model")
        test_dataset = TextDataset(X, label_dummy, wv_model)
        test_loader = get_dataloader(test_dataset,
                                     batch_size=kwargs.get("batch_size", 128),
                                     num_workers=kwargs.get("num_workers", 4),
                                     shuffle=True
                                    )
        y_pred = []
        for i, (data, label) in enumerate(test_loader):
            data = data.type(torch.FloatTensor)
            data = data.to(device)
            output = self.forward(data)
            _, idx = torch.max(output, dim=1)
            y_pred.extend(idx.tolist())
                
        return y_pred

    def get_X(self, sentence_list, **kwargs):
        return sentence_list
    
    def save_corpus(self, corpus_name, X, Y):
        if not os.path.exists("pkl_files/"):
            os.mkdir("pkl_files")
        n_file = "pkl_files/nn_" + corpus_name + ".pkl"
        pkl.dump([X, Y, self.labeler], open(n_file, "wb"))

    def load_corpus(self, corpus_name):
        n_file = "pkl_files/nn_" + corpus_name + ".pkl"
        bundle = pkl.load(open(n_file, "rb"))
        X, Y, self.labeler = bundle
        return X, Y
    
    def save_model(self, corpus_name):
        if not os.path.exists("pkl_files/"):
            os.mkdir("pkl_files")
        n_file = "pkl_files/nn_" + corpus_name + "_model.pkl"
        torch.save(self.state_dict(), n_file)
        params = {
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "num_layers": self.num_layers
        }
        p_file = "pkl_files/nn_" + corpus_name + "_params.pkl"
        pkl.dump(params, open(p_file, "wb"))
    
    def load_model(self, corpus_name):
        n_file = "pkl_files/nn_" + corpus_name + "_model.pkl"
        self.load_state_dict(torch.load(n_file))

        
def get_dataloader(dataset, **kwargs):
    params = {'batch_size': kwargs.get('batch_size', 128),
              'shuffle': kwargs.get('shuffle', False),
              'num_workers': kwargs.get('num_workers', 4)}
    text_generator = DataLoader(dataset, **params)
    return text_generator
