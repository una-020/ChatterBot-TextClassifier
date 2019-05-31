from sklearn.model_selection import StratifiedKFold
from gensim.models import KeyedVectors
from model import *
import utils
import argparse

def indexer(data, index_list):
    return [data[i] for i in index_list]


def kfold(sentences, labels, model_type, C=1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_type == 'nn':
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
            model = Logistic(C)
        else:
            model = Lstm(
                embed_dim=default['embed_dim'],
                num_classes=num_labels,
                wv_model=wv_model,
                hidden_dim=default['hidden_dim'],
                dropout=default['dropout'],
                num_layers=default['num_layers']
            )
            model.to(device)
    
        trainX = model.get_X(train_sent, fit=True)
        trainY = model.get_Y(train_label, fit=True)
        model.train_model(trainX, trainY)
        
        testX = model.get_X(test_sent, fit=False)
        testY = model.get_Y(test_label, fit=False)
        predictY = model.predict(testX, wv_model=wv_model)

        _, _, _, acc = model.eval_model(testY, predictY)
        print("Evaluated Split " + str(avg_cnt + 1) + " Accuracy: " + str(acc))
        avg_acc += acc
        avg_cnt += 1

    print("\nLook! K-fold results!!")
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
