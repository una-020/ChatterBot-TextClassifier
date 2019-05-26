from parsers.turksParser import parseLabel as parseLabelTurks
from parsers.turksParser import parseData as parseDataTurks
from parsers.uciParser import parseLabel as parseLabelUCI
from parsers.uciParser import parseData as parseDataUCI


def get_categories(total_labels):
    for index in range(len(total_labels)):
        if total_labels[index] == 'm':
            total_labels[index] = 'Health'
        elif total_labels[index] == 'SciTech' or total_labels[index] == 't':
            total_labels[index] = 'SciTech'
        elif total_labels[index] == 'Business' or total_labels[index] == 'b':
            total_labels[index] = 'Business'
        elif total_labels[index] == 'e':
            total_labels[index] = 'Entertainment'
    return total_labels


def parseData(uci_path, turks_path):
    sents_uci = parseDataUCI(uci_path)
    sents_turks = parseDataTurks(turks_path)
    return sents_turks + sents_uci


def parseLabel(uci_path, turks_path):
    labels_uci = parseLabelUCI(uci_path)
    labels_turks = parseLabelTurks(turks_path)
    total_labels = labels_turks + labels_uci
    total_labels = get_categories(total_labels)
    return total_labels


def main():
    uci_path = "data/news/uci.csv"
    turks_path = "data/news/turks.json"
    sents = parseData(uci_path, turks_path)
    labels = parseLabel(uci_path, turks_path)
    print(len(sents), len(labels))


if __name__ == "__main__":
    main()