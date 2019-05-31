from gensim.models import KeyedVectors
from model import recover_model
from utils import *

import sys
import argparse
import pickle as pkl


def predictor(text, model_name, corpus_name, wv_model=None):
    model, _, _ = recover_model(model_name, corpus_name, wv_model)
    
    testX = model.get_X(text, fit=False)
    y_pred = model.predict(testX, wv_model=wv_model)
    results = model.labeler.inverse_transform(y_pred)
    
    return results

        
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
    parser.add_argument('-ipath', '--input_path',
                        help='Input path',
                        type=str,
                        required=True)
    parser.add_argument('-opath', '--output_path',
                        help='Output path',
                        type=str,
                        default="predict.txt",
                        required=False)
    args = parser.parse_args()
    
    sentences = open(args.input_path).readlines()
    sentences = [line.strip() for line in sentences if line.strip()]
    sentences = preprocess_sentences(sentences)
    wv_model = None
    
    if args.model == 'nn':
        filename = 'GoogleNews-vectors-negative300.bin'
        wv_model = KeyedVectors.load_word2vec_format(filename, binary=True)
        print("WVEC loaded")
    
    predictions = predictor(sentences, args.model, args.corpus_name, wv_model)
    f = open(args.output_path, "w")
    for res in predictions:
        f.write(res + "\n")
    f.close()


if __name__ == "__main__":
    main()
