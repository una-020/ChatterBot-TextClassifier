import pandas as pd


def parseData(corpus_path):
    df = pd.read_csv(corpus_path)
    sentences = []
    for i in range(len(df)):
        sentences.append(df["TITLE"][i])
    return sentences


def parseLabel(corpus_path):
    df = pd.read_csv(corpus_path)
    labels = []
    for i in range(len(df)):
        labels.append(df["CATEGORY"][i])
    return labels


def main():
    file_path = "data/news/uci.csv"
    sents = parseData(file_path)
    labels = parseLabel(file_path)
    print(len(sents), len(labels))


if __name__ == "__main__":
    main()
