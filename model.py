from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from utils import *
import torch.nn as nn


class Model:
    def train(self, X, Y, **kwargs):
        pass

    def predict(self, X, **kwargs):
        pass

    def get_X(self, sentence_list, transform=True):
        pass

    def get_Y(self, label_list, transform=True):
        if transform:
            self.labeler = preprocessing.LabelEncoder()
            Y = self.labeler.fit_transform(label_list)
        else:
            Y = self.labeler.fit(label_list)
        return Y

    def eval(self, y_true, y_pred, average):
        assert len(y_true) == len(y_pred)
        precision, recall, fscore, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=average
        )
        accuracy = accuracy_score(y_true, y_pred)
        return precision, recall, fscore, accuracy


class Logistic(Model):
    def __init__(self, C):
        self.cls = LogisticRegression(
            random_state=0,
            C=C,
            solver='lbfgs',
            max_iter=10000,
            n_jobs=-1,
            class_weight='balanced'
        )

    def train(self, X, Y, **kwargs):
        self.cls.fit(X, Y)

    def predict(self, X, **kwargs):
        return self.cls.predict(X)

    def get_X(self, sentence_list, transform=True):
        if transform:
            sentence_list = preprocess_periods(sentence_list)
            self.vect = TfidfVectorizer(
                tokenizer=word_tokenize,
            )
            X = self.vect.fit_transform(sentence_list)
        else:
            X = self.vect.fit(sentence_list)
        return X
