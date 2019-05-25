from sklearn.model_selection import StratifiedKFold
from model import *
import argparse


def indexer(data, index_list):
    return [data[i] for i in index_list]


def kfold(sentences, labels, model_type, C=1):
    avg_acc, avg_cnt = 0.0, 0.0
    kf = StratifiedKFold(n_splits=5, shuffle=True)

    for train_index, test_index in kf.split(sentences, labels):
        train_sent = indexer(sentences, train_index)
        test_sent = indexer(sentences, test_index)
        train_label = indexer(labels, train_index)
        test_label = indexer(labels, test_index)

        if model_type == "lr":
            model = Logistic(C)

        trainX = model.get_X(train_sent, fit=True)
        trainY = model.get_Y(train_label, fit=True)
        model.train(trainX, trainY)

        testX = model.get_X(test_sent, fit=False)
        testY = model.get_Y(test_label, fit=False)
        predictY = model.predict(testX)

        print("Evaluating Split", avg_cnt + 1)
        _, _, _, acc = model.eval(testY, predictY)
        avg_acc += acc
        avg_cnt += 1

    print("Average Accuracy", avg_acc / avg_cnt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='(lr|nn)',
                        required=True, type=str)
    parser.add_argument('-C', '--C', help='Inverse lr reg. strength',
                        required=False, type=float, default=1)
    parser.add_argument('-cname', '--corpus_name',
                        help='Corpus name', required=True, type=str)
    args = parser.parse_args()
    sentences = get_corpus(args.corpus_name)
    labels = get_label(args.corpus_name)
    kfold(sentences, labels, args.model, args.C)


if __name__ == "__main__":
    main()
