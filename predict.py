from utils import *
from dataloader import *
import sys
import argparse
import pickle as pkl


def predictor(text, model_name, corpus_name, wv_model=None):
    model, _, _ = recover_model(model_name, corpus_name, wv_model)
    
    testX = model.get_X(text, fit=False)
    y_pred = model.predict(testX, wv_model)
    results = model.labeler.inverse_transform(y_pred)
    
    f = sys.stdout
    for res in results:
        f.write(res + "\n")        

        
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
    wv_model = None
    
    if args.model == 'nn':
        filename = 'GoogleNews-vectors-negative300.bin'
        wv_model = KeyedVectors.load_word2vec_format(filename, binary=True)
        print("WVEC loaded")
        
    predictor(text, args.model, args.corpus_name, wv_model)

if __name__ == "__main__":
    main()
