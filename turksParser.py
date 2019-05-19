import json


def parseData(corpus_path):
    sentences = []
    for data_line in open(corpus_path):
        data = json.loads(data_line)
        sentences.append(data["content"])
    return sentences


def parseLabel(corpus_path):
    labels = []
    for data_line in open(corpus_path):
        data = json.loads(data_line)
        labels.append(data["annotation"]["label"][0])
    return labels


def main():
    file_path = "data/news/turks.json"
    sents = parseData(file_path)
    labels = parseLabel(file_path)
    print(len(sents), len(labels))


if __name__ == "__main__":
    main()
