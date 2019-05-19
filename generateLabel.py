from sklearn import preprocessing
import sys
import pickle as pkl


def getYBasic(labels):
    labeler = preprocessing.LabelEncoder()
    Y = labeler.fit_transform(labels)
    return Y, labeler


def main():
    assert len(sys.argv) in [1, 2], "Supply dataset type [uci/turks/sarcasm/blog/sentiment], field name [TODO]"

    if sys.argv[1] == "uci":
        from uciParser import parseLabel

        labels = parseLabel("data/news/uci.csv")
    elif sys.argv[1] == "turks":
        from turksParser import parseLabel

        labels = parseLabel("data/news/turks.json")
    elif sys.argv[1] == "sarcasm":
        from sarcasmParser import parseLabel

        labels = parseLabel("data/news/sarcasm.json")
    elif sys.argv[1] == "blog":
        from blogParser import parseLabel

        if len(sys.argv) == 1:
            labels = parseLabel("data/blog/TODO")
        elif len(sys.argv) == 2:
            labels = parseLabel("data/blog/TODO", sys.argv[2])
    elif sys.argv[1] == "sentiment":
        from sentimentParser import parseLabel

        labels = parseLabel("data/sentiment/sentiment.csv")
    Y, labeler = getYBasic(labels)
    print(len(labels))
    print(Y.shape, labeler.classes_)
    pkl.dump(Y, open("Y.pkl", "wb"))


if __name__ == "__main__":
    main()
