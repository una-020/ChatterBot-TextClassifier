from gensim.models import KeyedVectors
from model import *
from utils import *
import argparse
import pickle as pkl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        help='(lr|nn)',
                        type=str,
                        required=True)
    parser.add_argument('-cname', '--corpus_name',
                        help='Corpus name',
                        type=str,
                        required=True)
    parser.add_argument('-C', '--C',
                        help='Inverse lr reg. strength',
                        type=float,
                        default=default["C"],
                        required=False)
    parser.add_argument('-edim', '--embed_dim',
                        help='Word embedding dimension',
                        type=int,
                        default=default["embed_dim"],
                        required=False)
    parser.add_argument('-hdim', '--hidden_dim',
                        help='Hidden Dimension',
                        type=int,
                        default=default["hidden_dim"],
                        required=False)
    parser.add_argument('-drop', '--dropout',
                        help='Dropout in NN',
                        type=float,
                        default=default["dropout"],
                        required=False)
    parser.add_argument('-nlayers', '--num_layers',
                        help='Number of layers in NN',
                        type=int,
                        default=default["num_layers"],
                        required=False)
    parser.add_argument('-e', '--epoch',
                        help='Epoch',
                        type=int,
                        default=default["epoch"],
                        required=False)
    parser.add_argument('-bsize', '--batch_size',
                        help='Batch Size',
                        type=int,
                        default=default["batch_size"],
                        required=False)
    parser.add_argument('-nworkers', '--num_workers',
                        help='Number of worker threads',
                        type=int,
                        default=default["num_workers"],
                        required=False)
    parser.add_argument('-r', '--recover', action='store_true')

    args = parser.parse_args()

    sentences = get_corpus(args.corpus_name)
    sentences = preprocess_sentences(sentences)
    labels = get_label(args.corpus_name)
    num_classes = len(set(labels))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    if args.recover:
        if args.model == "lr":
            wv_model = None
        else:
            filename = 'GoogleNews-vectors-negative300.bin'
            wv_model = KeyedVectors.load_word2vec_format(filename, binary=True)
        model, X, Y = recover_model(args.model, args.corpus_name, wv_model)
        print("Recovered model")
        print(model)
    else:
        if args.model == "lr":
            model = Logistic(args.C)
            wv_model = None
        else:
            filename = 'GoogleNews-vectors-negative300.bin'
            wv_model = KeyedVectors.load_word2vec_format(filename, binary=True)
            print("WVEC loaded")
            model = Lstm(
                embed_dim=args.embed_dim,
                num_classes=num_classes,
                wv_model=wv_model,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                num_layers=args.num_layers
            )
            model = model.to(device)
        print("New model")
        print(model)
        
        try:
            X, Y = model.load_corpus(args.corpus_name)
        except:
            X = model.get_X(sentences, fit=True)
            Y = model.get_Y(labels, fit=True)
            model.save_corpus(args.corpus_name, X, Y)

    model.train_model(X, Y, batch_size=args.batch_size, num_workers=args.num_workers, epoch=args.epoch)
    model.save_model(args.corpus_name)


if __name__ == "__main__":
    main()
