def preprocess_periods(sentences):
    new_sentences = []
    for sentence in sentences:
        new_sentence = " ".join(sentence.split("."))
        new_sentence = new_sentence.lower()
        new_sentences.append(new_sentence)
    return new_sentences

def get_corpus(corpus_name):
    if corpus_name == "uci":
        from uciParser import parseData
        sentences = parseData("data/news/uci.csv")

    elif corpus_name == "turks":
        from turksParser import parseData
        sentences = parseData("data/news/turks.json")

    elif corpus_name == "news":
        from newsParser import parseData
        sentences = parseData("data/news/combined_news.csv")

    elif corpus_name == "sarcasm":
        from sarcasmParser import parseData
        sentences = parseData("data/news/sarcasm.json")

    elif corpus_name == "blog":
        from blogParser import parseData
        sentences = parseData("data/blog/blogs.txt")

    elif corpus_name == "sentiment":
        from sentimentParser import parseData
        sentences = parseData("data/sentiment/sentiment.csv")

    return sentences


def get_label(corpus_name, category="gender"):
    if corpus_name == "uci":
        from uciParser import parseLabel
        labels = parseLabel("data/news/uci.csv")

    elif corpus_name == "turks":
        from turksParser import parseLabel
        labels = parseLabel("data/news/turks.json")

    elif corpus_name == "news":
        from newsParser import parseLabel
        labels = parseLabel("data/news/combined_news.csv")

    elif corpus_name == "sarcasm":
        from sarcasmParser import parseLabel
        labels = parseLabel("data/news/sarcasm.json")

    elif corpus_name == "blog":
        from blogParser import parseLabel
        labels = parseLabel("data/blog/{}.txt".format(category))

    elif corpus_name[1] == "sentiment":
        from sentimentParser import parseLabel
        labels = parseLabel("data/sentiment/sentiment.csv")

    return labels
