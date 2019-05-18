import json


def parseData(corpus_path):
    sentences = []
    for data_line in open(corpus_path):
        data = json.loads(data_line)
        sentences.append(data["headline"])
    return sentences


def parseLabel(corpus_path, field):
    assert field in ["is_sarcastic"]
    labels = []
    for data_line in open(corpus_path):
        data = json.loads(data_line)
        labels.append(data[field])
    return labels


def main():
    file_path = "news/sarcasm.json"
    sents = parseData(file_path)
    labels = parseLabel(file_path, "is_sarcastic")
    print(len(sents), len(labels))


if __name__ == "__main__":
    main()
