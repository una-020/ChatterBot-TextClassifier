from urllib.parse import urlparse, urlencode
from urllib.request import urlopen, Request
from gensim.models import KeyedVectors
from urllib.error import HTTPError
from flask import make_response
from model import recover_model
from explain import explainer
from predict import predictor
from flask import request
from flask import Flask

import os
import json


# Flask app should start in global layout
app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    global corpus
    global sentence
    global explain_corpus
    global explain_sentence
    global model
    global wv_model

    req = request.get_json(silent=True, force=True)
    intent = req['queryResult']['intent']['displayName']
    textResponse = req['queryResult']['fulfillmentText']
    if intent == 'CorpusSelect - custom':
        corpus = req['queryResult']['parameters']['corpus']
    elif intent == 'SentenceInput - custom':
        sentence = req['queryResult']['parameters']['any']
    elif intent == 'Predict':
        if corpus is None or sentence is None:
            textResponse = "Choose the corpus and the sentence first."
        else:
            prediction = predictor([sentence], model[corpus], wv_model)[0]
            textResponse = "The sentence was processed by my Brain :) and the category is (drumroll) %s" % prediction
    elif intent == 'Explain':
        if corpus is None or sentence is None:
            textResponse = "Choose the corpus and the sentence first."
        elif explain_corpus == corpus and explain_sentence == sentence:
            if os.path.exists("static/explain.html"):
                textResponse = "Find my explanations here:\nhttps://chatterbot.serveo.net/static/explain.html"
            else:
                pass
        else:
            explain_corpus = corpus
            explain_sentence = sentence
            if not os.path.exists("static/"):
                os.mkdir("static/")
            elif os.path.exists("static/explain.html"):
                os.remove("static/explain.html")
            explain_html = explainer(sentence, model[corpus], wv_model)
            f = open("static/explain.html", "w")
            f.write(explain_html)
            f.close()
    else:
        print("DEBUG:", req)
    
    res_dict = {"fulfillmentText": textResponse, "source": "webhook"}
    res_json = json.dumps(res_dict, indent=4)
    res = make_response(res_json)
    res.headers['Content-Type'] = 'application/json'
    return res


if __name__ == '__main__':
    global wv_model
    filename = 'GoogleNews-vectors-negative300.bin.gz'
    wv_model = KeyedVectors.load_word2vec_format(filename, binary=True)
    # wv_model = None
    print("WVEC Loaded")
    
    global model
    model = {}
    model['turks'], _, _ = recover_model('nn', 'turks', wv_model)
    model['news'], _, _ = recover_model('nn', 'news', wv_model)
    model['uci'], _, _ = recover_model('nn', 'uci', wv_model)
    model['sentiment'], _, _ = recover_model('nn', 'sentiment', wv_model)

    global corpus
    corpus = None
    
    global sentence
    sentence = None
    
    global explain_corpus
    explain_corpus = None
    
    global explain_sentence
    explain_sentence = None
    
    print('Recovered Models')
    
    port = int(os.getenv('PORT', 5000))
    print("Starting app on port %d" % port)
    app.run(port=port, host='0.0.0.0')
