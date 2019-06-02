from generateFeature import preprocess_periods
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from gensim.models import KeyedVectors
from model import recover_model
from utils import *

import sys
import argparse
import pickle as pkl


def explainer(text, model_name, corpus_name, wv_model=None, **kwargs):
#     text is a single sentence
    model, _, _ = recover_model(model_name, corpus_name, wv_model)
    
    def predict_proba(sentences):
        testX = model.get_X(sentences, fit=False)
        if model_name == "lr":
            return model.cls.predict_proba(testX)
        else:
            return model.predict_proba(testX, wv_model=wv_model)

    explainer = LimeTextExplainer(class_names=model.labeler.classes_)
    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=kwargs.get("num_features", 6),
        top_labels=kwargs.get("top_labels", 2)
    )
    return exp.as_html(text=text)


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
                        default="explain.html",
                        required=False)
    args = parser.parse_args()
    
    sentence = open(args.input_path).read().strip()
    wv_model = None
    
    if args.model == 'nn':
        filename = 'GoogleNews-vectors-negative300.bin'
        wv_model = KeyedVectors.load_word2vec_format(filename, binary=True)
        print("WVEC loaded")
    
    f = open(args.output_path, "w")
    f.write(explainer(sentence, args.model, args.corpus_name, wv_model))
    f.close()


if __name__ == "__main__":
    main()
