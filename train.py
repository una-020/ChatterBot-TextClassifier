from model import *
from utils import *
import argparse
import pickle as pkl


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

    if args.model == "lr":
        model = Logistic(args.C)

    try:
        X = pkl.load(
            open(args.model + "_" + args.corpus_name + "_X.pkl", "rb")
        )
        Y = pkl.load(
            open(args.model + "_" + args.corpus_name + "_Y.pkl", "rb")
        )
    except FileNotFoundError:
        X = model.get_X(sentences, fit=True)
        Y = model.get_Y(labels, fit=True)
        pkl.dump(
            X,
            open(args.model + "_" + args.corpus_name + "_X.pkl", "wb")
        )
        pkl.dump(
            Y,
            open(args.model + "_" + args.corpus_name + "_Y.pkl", "wb")
        )

    model.train(X, Y)
    pkl.dump(
        model,
        open(args.model + "_" + args.corpus_name + "_mod.pkl", "wb")
    )


if __name__ == "__main__":
    main()
