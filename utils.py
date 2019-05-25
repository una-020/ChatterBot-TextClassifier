import pickle as pkl


def preprocess_periods(sentences):
    new_sentences = []
    for sentence in sentences:
        new_sentence = " ".join(sentence.split("."))
        new_sentence = new_sentence.lower()
        new_sentences.append(new_sentence)
    return new_sentences


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
        sentences = parseData("data/news/combined_news.csv")

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
        labels = parseLabel("data/news/combined_news.csv")

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
        open(model_name + "_" + corpus_name + "_X.pkl", "rb")
    )
    Y = pkl.load(
        open(model_name + "_" + corpus_name + "_Y.pkl", "rb")
    )
    model.vect = pkl.load(
        open(model_name + "_" + corpus_name + "_X_vect.pkl", "rb")
    )
    model.labeler = pkl.load(
        open(model_name + "_" + corpus_name + "_Y_le.pkl", "rb")
    )
    return X, Y


def save_features_lr(X, Y, model, model_name, corpus_name):
    pkl.dump(
        X,
        open(model_name + "_" + corpus_name + "_X.pkl", "wb")
    )
    pkl.dump(
        Y,
        open(model_name + "_" + corpus_name + "_Y.pkl", "wb")
    )
    pkl.dump(
        model.vect,
        open(model_name + "_" + corpus_name + "_X_vect.pkl", "wb")
    )
    pkl.dump(
        model.labeler,
        open(model_name + "_" + corpus_name + "_Y_le.pkl", "wb")
    )


def get_max_len(sentence_list):
    max_len = 0
    for sentence in sentence_list:
        if len(sentence) > max_len:
            max_len = len(sentence)
    return max_len
