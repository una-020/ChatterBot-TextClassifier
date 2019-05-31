from utils import *

import sys
import argparse
import pickle as pkl


def predictor(text, model_name, corpus_name, wv_model=None):
    labels = get_label(corpus_name)
    num_classes = len(set(labels))
    
    p_file = "pkl_files/" + model_name + "_" + corpus_name + "_params.pkl"
    params = pkl.load(open(p_file, "rb"))
    C = params.get("C", default["C"])
    embed_dim = params.get("embed_dim", default["embed_dim"])
    hidden_dim = params.get("hidden_dim", default["hidden_dim"])
    dropout = params.get("dropout", default["dropout"])
    num_layers = params.get("num_layers", default["num_layers"])
    num_classes = params.get("num_classes", num_classes)
    
    if args.model == "lr":
        model = Logistic(C)
        model.load_model(args.corpus_name)
        model.lo
    else:
        model = Lstm(
            embed_dim=embed_dim,
            num_classes=num_classes,
            wv_model=wv_model,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            num_layers=args.num_layers
        )
        if args.recover:
            model.load_model(args.corpus_name)
        model = model.to(device)
    if model_name == "lr":
        

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
    args = parser.parse_args()
    
    text = input().strip()
    text = preprocess_sentences([text])[0]
    
    p_file = "pkl_files/" + args.model + "_" + args.corpus_name + "_params.pkl"
    params = pkl.load(open(p_file, "rb"))
    args.C = params.get("C", default["C"])
    args.embed_dim = params.get("embed_dim", default["embed_dim"])
    args.hidden_dim = params.get("hidden_dim", default["hidden_dim"])
    args.dropout = params.get("dropout", default["dropout"])
    args.num_layers = params.get("num_layers", default["num_layers"])
    num_classes = params.get("num_classes", num_classes)
        print("Recovering...")

    model = pkl.load(open(args.model_path, "rb"))
    sentences = [line.strip() for line in open(args.input_path).readlines()]
    sentences = [line for line in sentences if len(line) > 0]

    testX = model.get_X(sentences, fit=False)
    y_pred = model.predict(testX)
    results = model.labeler.inverse_transform(y_pred)

    f = sys.stdout
    for res in results:
        f.write(res + "\n")


if __name__ == "__main__":
    main()
