from torch.utils.data import DataLoader
from gensim.models import KeyedVectors
from dataloader import *
from model import *
import os
import re
import torch
import pickle as pkl


default = {
    "C": 1,
    "embed_dim": 300,
    "hidden_dim": 200,
    "dropout": 0.5,
    "num_layers": 2,
    "epoch": 100,
    "batch_size": 128,
    "num_workers": 4
}


def recover_model(model_name, corpus_name, wv_model=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    b_name = "pkl_files/" + model_name + "_" + corpus_name + ".pkl"
    bundle = pkl.load(open(b_name, "rb"))
    labeler = bundle[-1]
    X = bundle[0]
    Y = bundle[1]
    
    p_file = "pkl_files/" + model_name + "_" + corpus_name + "_params.pkl"
    params = pkl.load(open(p_file, "rb"))
    C = params.get("C", default["C"])
    embed_dim = params.get("embed_dim", default["embed_dim"])
    hidden_dim = params.get("hidden_dim", default["hidden_dim"])
    dropout = params.get("dropout", default["dropout"])
    num_layers = params.get("num_layers", default["num_layers"])
    num_classes = params.get("num_classes", len(labeler.classes_))
    
    if model_name == "lr":
        model = Logistic(C)
        model.labeler = labeler
        model.vect = bundle[-2]
        model.load_model(corpus_name)
    else:
        assert wv_model is not None
        model = Lstm(
            embed_dim=embed_dim,
            num_classes=num_classes,
            wv_model=wv_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_layers
        )
        model.labeler = labeler
        model.load_model(corpus_name)
        model = model.to(device)
    return model, X, Y


def preprocess_sentences(sentences):
    return [re.sub('[^A-Za-z0-9]+', ' ', sent.lower()) for sent in sentences]


def get_corpus(corpus_name):
    corpus_name = corpus_name.split("-")[0]
    if corpus_name == "uci":
        from parsers.uciParser import parseData
        sentences = parseData("data/news/uci.csv")

    elif corpus_name == "turks":
        from parsers.turksParser import parseData
        sentences = parseData("data/news/turks.json")

    elif corpus_name == "news":
        from parsers.newsParser import parseData
        sentences = parseData("data/news/uci.csv", "data/news/turks.json")

    elif corpus_name == "sarcasm":
        from parsers.sarcasmParser import parseData
        sentences = parseData("data/news/sarcasm.json")

    elif corpus_name == "blog":
        from parsers.blogParser import parseData
        sentences = parseData("data/blog/blogs.txt")

    elif corpus_name == "sentiment":
        from parsers.sentimentParser import parseData
        sentences = parseData("data/sentiment/sentiment.csv")

    return sentences


def get_label(corpus_name):
    name_split = corpus_name.split("-")
    if len(name_split) == 2:
        category = name_split[1]
    corpus_name = name_split[0]

    if corpus_name == "uci":
        from parsers.uciParser import parseLabel
        labels = parseLabel("data/news/uci.csv")

    elif corpus_name == "turks":
        from parsers.turksParser import parseLabel
        labels = parseLabel("data/news/turks.json")

    elif corpus_name == "news":
        from parsers.newsParser import parseLabel
        labels = parseLabel("data/news/uci.csv", "data/news/turks.json")

    elif corpus_name == "sarcasm":
        from parsers.sarcasmParser import parseLabel
        labels = parseLabel("data/news/sarcasm.json")

    elif corpus_name == "blog":
        from parsers.blogParser import parseLabel
        labels = parseLabel("data/blog/{}.txt".format(category))

    elif corpus_name[1] == "sentiment":
        from parsers.sentimentParser import parseLabel
        labels = parseLabel("data/sentiment/sentiment.csv")

    return labels


def get_max_len(sentence_list):
    return max([len(sentence.split()) for sentence in sentence_list])


