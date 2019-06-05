from sklearn.model_selection import StratifiedKFold
from parsers.newsParser import get_categories
from gensim.models import KeyedVectors
from model import *
import argparse


def indexer(data, index_list):
    return [data[i] for i in index_list]


def kfold(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    sentences = get_corpus(args.corpus_name)
    labels = get_label(args.corpus_name)
    
    if args.model == 'nn':
        filename = 'GoogleNews-vectors-negative300.bin'
        wv_model = KeyedVectors.load_word2vec_format(filename, binary=True)
        print("WVEC loaded")
    else:
        wv_model = None
        
    avg_acc, avg_cnt = 0.0, 0.0
    num_labels = len(set(labels))
    kf = StratifiedKFold(n_splits=5, shuffle=True)

    for train_index, test_index in kf.split(sentences, labels):
        train_sent = indexer(sentences, train_index)
        test_sent = indexer(sentences, test_index)
        train_label = indexer(labels, train_index)
        test_label = indexer(labels, test_index)
        
        if model_type == "lr":
            model = Logistic(args.C)
        else:
            model = Lstm(
                embed_dim=args.embed_dim,
                num_classes=num_labels,
                wv_model=wv_model,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                num_layers=args.num_layers
            )
            model.to(device)
    
        trainX = model.get_X(train_sent, fit=True)
        trainY = model.get_Y(train_label, fit=True)
        model.train_model(trainX, trainY, batch_size=args.batch_size, num_workers=args.num_workers, epoch=args.epoch)
        
        testX = model.get_X(test_sent, fit=False)
        testY = model.get_Y(test_label, fit=False)
        predictY = model.predict(testX, wv_model=wv_model)

        _, _, _, acc = model.eval_model(testY, predictY)
        print("Evaluated Split " + str(avg_cnt + 1) + " Accuracy: " + str(acc))
        avg_acc += acc
        avg_cnt += 1

    print("\nLook! K-fold results!!")
    print("Average Accuracy", avg_acc / avg_cnt)

    
def mixed_models(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model == 'nn':
        filename = 'GoogleNews-vectors-negative300.bin'
        wv_model = KeyedVectors.load_word2vec_format(filename, binary=True)
        print("WVEC loaded")
    else:
        wv_model = None
    
    model, _, _ = recover_model(args.model, args.corpus_name, wv_model)
    
    sentences = get_corpus(args.test_corpus)
    labels = get_label(args.test_corpus)
    
    if args.corpus_name != args.test_corpus:
        labels = get_categories(labels)
   
    testX = model.get_X(sentences, fit=False)
    testY = model.get_Y(labels, fit=False)
    
    predictY = model.predict(testX, wv_model=wv_model)
    _, _, _, acc = model.eval_model(testY, predictY)
    print(" Accuracy: " + str(acc))


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
                        default=1,
                        required=False)
    parser.add_argument('-edim', '--embed_dim',
                        help='Word embedding dimension',
                        type=int,
                        default=300,
                        required=False)
    parser.add_argument('-hdim', '--hidden_dim',
                        help='Hidden Dimension',
                        type=int,
                        default=200,
                        required=False)
    parser.add_argument('-drop', '--dropout',
                        help='Dropout in NN',
                        type=float,
                        default=0.5,
                        required=False)
    parser.add_argument('-nlayers', '--num_layers',
                        help='Number of layers in NN',
                        type=int,
                        default=2,
                        required=False)
    parser.add_argument('-e', '--epoch',
                        help='Epoch',
                        type=int,
                        default=100,
                        required=False)
    parser.add_argument('-bsize', '--batch_size',
                        help='Batch Size',
                        type=int,
                        default=128,
                        required=False)
    parser.add_argument('-nworkers', '--num_workers',
                        help='Number of worker threads',
                        type=int,
                        default=4,
                        required=False)
    
    parser.add_argument('-option', '--option',
                        help='kfold | test',
                        type=str,
                        default='kfold',
                        required=False)

    parser.add_argument('-test_corpus', '--test_corpus',
                        help='name of the corpus you want to test on',
                        type=str,
                        default='turks',
                        required=False)
 
    args = parser.parse_args()
    
    if args.option == 'kfold':
        kfold(args)
    else:
        mixed_models(args)

if __name__ == "__main__":
    main()
