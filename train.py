from gensim.models import KeyedVectors
from model import *
from utils import *
import argparse
import pickle as pkl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='(lr|nn)',
                        required=True, type=str)
    parser.add_argument('-C', '--C', help='Inverse lr reg. strength',
                        required=False, type=float, default=1.0)
    parser.add_argument('-cname', '--corpus_name',
                        help='Corpus name', required=True, type=str)
    parser.add_argument('-em_dim', '--embed_dim',
                        help='Word embedding dimension', type=int, default=300)
    parser.add_argument('-b_size', '--batch_size',
                        help='Batch Size', type=int, default=128)
    parser.add_argument('-num_workers', '--num_workers',
                        help='Number of worker threads', type=int, default=4)

    args = parser.parse_args()

    sentences = get_corpus(args.corpus_name)
    labels = get_label(args.corpus_name)

    if args.model == "lr":
        model = Logistic(args.C)
        wv_model = None

    else:
        filename = 'embed.bin'
        wv_model = KeyedVectors.load_word2vec_format(filename, binary=True)
        model = Lstm(embed_dim=args.embed_dim, num_classes=len(set(labels)))

    try:
        if args.model == "lr":
            X, Y = get_features_lr(
                model, args.model, args.corpus_name
            )

    except FileNotFoundError:
        if args.model == "lr":
            X = model.get_X(sentences, fit=True, wv_model=wv_model)
            Y = model.get_Y(labels, fit=True)
            save_features_lr(X, Y, model, args.model, args.corpus_name)

    model.train(X, Y, batch_size=args.batch_size, num_workers=args.num_workers)
    pkl.dump(
        model,
        open('pkl_files/' + args.model + "_" + args.corpus_name + "_mod.pkl", "wb")
    )


if __name__ == "__main__":
    main()
