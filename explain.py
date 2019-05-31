from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from generateFeature import preprocess_periods
import pickle as pkl
import sys


def explain(text, model, vect, labeler):
    c = make_pipeline(vect, model)
    explainer = LimeTextExplainer(class_names=labeler.classes_)
    exp = explainer.explain_instance(
        text,
        c.predict_proba,
        num_features=6,
        top_labels=2
    )
    return exp.as_html(text=text)


def main():
    text = input().strip()
    text = preprocess_sentences([text])[0]
    vect = pkl.load(open("Xvect.pkl", "rb"))
    labeler = pkl.load(open("Ylabeler.pkl", "rb"))
    cls = pkl.load(open("model.pkl", "rb"))

    f = sys.stdout
    f.write(explain(text, cls, vect, labeler))


if __name__ == "__main__":
    main()
