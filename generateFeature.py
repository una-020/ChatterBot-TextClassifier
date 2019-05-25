from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import sys
import pickle as pkl


def preprocess_periods(sentences):
    new_sentences = []
    for sentence in sentences:
        new_sentence = " ".join(sentence.split("."))
        new_sentence = new_sentence.lower()
        new_sentences.append(new_sentence)
    return new_sentences


def getXBasic(sentences):
    sentences = preprocess_periods(sentences)
    vect = TfidfVectorizer(
        tokenizer=word_tokenize,
    )
    X = vect.fit_transform(sentences)
    return X, vect


def main():
    assert len(sys.argv) == 2, "Supply dataset type [uci/turks/sarcasm/blog/sentiment]"

    if sys.argv[1] == "uci":
        from uciParser import parseData
        sentences = parseData("data/news/uci.csv")

    elif sys.argv[1] == "turks":
        from turksParser import parseData
        sentences = parseData("data/news/turks.json")

    elif sys.argv[1] == "news":
        from newsParser import parseData
        sentences = parseData("data/news/combined_news.csv")

    elif sys.argv[1] == "sarcasm":
        from sarcasmParser import parseData
        sentences = parseData("data/news/sarcasm.json")

    elif sys.argv[1] == "blog":
        from blogParser import parseData
        sentences = parseData("data/blog/blogs.txt")

    elif sys.argv[1] == "sentiment":
        from sentimentParser import parseData
        sentences = parseData("data/sentiment/sentiment.csv")

    X, vect = getXBasic(sentences)
    print(len(sentences), "sentences found!")
    print(X.shape)
    pkl.dump(X, open("X.pkl", "wb"))
    pkl.dump(vect, open("Xvect.pkl", "wb"))


if __name__ == "__main__":
    main()
