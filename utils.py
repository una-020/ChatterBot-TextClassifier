from torch.utils.data import DataLoader
from dataloader import *
import os
import re
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

    elif corpus_name == "sentiment":
        from parsers.sentimentParser import parseLabel
        labels = parseLabel("data/sentiment/sentiment.csv")

    return labels


def get_max_len(sentence_list):
    return max([len(sentence.split()) for sentence in sentence_list])


def get_dataloader(dataset, **kwargs):
    params = {'batch_size': kwargs.get('batch_size', default["batch_size"]),
              'shuffle': kwargs.get('shuffle', False),
              'num_workers': kwargs.get('num_workers', default["num_workers"])}
    text_generator = DataLoader(dataset, **params)
    return text_generator
