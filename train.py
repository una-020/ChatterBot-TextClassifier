from model import *
from utils import *
import sys
import argparse
import pickle as pkl


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m', '--model', help='(lr|nn)', required=True, type=str)
    parser.add_argument('-C', '--C', help='Inverse lr reg. strength', required=False, type=float, default=1)
    parser.add_argument('-cname', '--corpus_name', help='Corpus name', required=True, type=str)
    parser.add_argument('-ccat', '--corpus_category', help='Corpus category', required=False, type=str, default="gender")
    args = parser.parse_args()

    sentences = get_corpus(args.corpus_name)
    labels = get_label(args.corpus_name, category=args.corpus_category)

    if args.model == "lr":
        model = Logistic(args.C)
    
    X = model.get_X(sentences, fit=True)
    Y = model.get_Y(labels, fit=True)
    model.train(X, Y)
    pkl.dump(model, open(args.model + "_" + args.corpus_name + "_mod.pkl", "wb"))

    y_pred = model.predict(X)
    print(model.eval(Y, y_pred, average=None))


if __name__ == "__main__":
    main()
