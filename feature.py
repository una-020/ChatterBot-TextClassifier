from sklearn import preprocessing
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import pickle as pkl


def preprocess_periods(sentences):
    new_sentences = []
    for sentence in sentences:
        new_sentence = " ".join(sentence.split("."))
        new_sentences.append(new_sentence)
    return new_sentences


def getXYBasic(sentences, labels):
    sentences = preprocess_periods(sentences)
    vect = TfidfVectorizer(
        tokenizer=word_tokenize,
    )
    X = vect.fit_transform(sentences)
    labeler = preprocessing.LabelEncoder()
    Y = labeler.fit_transform(labels)
    return X, Y, vect, labeler


def main():
    assert len(sys.argv) == 2, "Supply dataset type"

    if sys.argv[1] == "uci":
        from uciParser import parseData
        from uciParser import parseLabel

        sentences = parseData("news/uci.csv")
        labels = parseLabel("news/uci.csv", "CATEGORY")
    elif sys.argv[1] == "turks":
        from turksParser import parseData
        from turksParser import parseLabel

        sentences = parseData("news/turks.json")
        labels = parseLabel("news/turks.json", "label")
    elif sys.argv[1] == "sarcasm":
        from sarcasmParser import parseData
        from sarcasmParser import parseLabel

        sentences = parseData("news/sarcasm.json")
        labels = parseLabel("news/sarcasm.json", "is_sarcastic")

    X, Y, vect, labeler = getXYBasic(sentences, labels)
    print(len(sentences), len(labels))
    print(X.shape, Y.shape, labeler.classes_)
    pkl.dump(X, open("X.pkl", "wb"))
    pkl.dump(Y, open("Y.pkl", "wb"))


if __name__ == "__main__":
    main()
