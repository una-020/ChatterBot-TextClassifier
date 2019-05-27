from torch.utils.data import DataLoader
from dataloader import *
import os
import re
import pickle as pkl


def preprocess_periods(sentences):
    new_sentences = []
    for sentence in sentences:
        new_sentence = " ".join(sentence.split("."))
        new_sentence = new_sentence.lower()
        new_sentences.append(new_sentence)
    return new_sentences


def preprocess_sentence(sentences):
    sentences = preprocess_periods(sentences)
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


def get_features_lr(model, model_name, corpus_name):
    X = pkl.load(
        open('pkl_files/' + model_name + "_" + corpus_name + "_X.pkl", "rb")
    )
    Y = pkl.load(
        open('pkl_files/' + model_name + "_" + corpus_name + "_Y.pkl", "rb")
    )
    model.vect = pkl.load(
        open('pkl_files/' + model_name + "_" + corpus_name + "_X_vect.pkl", "rb")
    )
    model.labeler = pkl.load(
        open('pkl_files/' + model_name + "_" + corpus_name + "_Y_le.pkl", "rb")
    )
    return X, Y


def save_features_lr(X, Y, model, model_name, corpus_name):
    if not os.path.exists("pkl_files/"):
        os.mkdir("pkl_files") 

    pkl.dump(
        X,
        open('pkl_files/' + model_name + "_" + corpus_name + "_X.pkl", "wb")
    )
    pkl.dump(
        Y,
        open('pkl_files/' + model_name + "_" + corpus_name + "_Y.pkl", "wb")
    )
    pkl.dump(
        model.vect,
        open('pkl_files/' + model_name + "_" + corpus_name + "_X_vect.pkl", "wb")
    )
    pkl.dump(
        model.labeler,
        open('pkl_files/' + model_name + "_" + corpus_name + "_Y_le.pkl", "wb")
    )


def get_max_len(sentence_list):
    return max([len(sentence.split()) for sentence in sentence_list])


def get_dataloader(X, Y, **kwargs):
    params = {'batch_size': kwargs.get('batch_size', 128),
              'shuffle': kwargs.get('shuffle', True),
              'num_workers': kwargs.get('num_workers', 16)}

    textdataset = TextDataset(X, Y)
    text_generator = DataLoader(textdataset, **params)

    return text_generator
