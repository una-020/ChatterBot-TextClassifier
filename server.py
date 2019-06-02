from urllib.parse import urlparse, urlencode
from urllib.request import urlopen, Request
from gensim.models import KeyedVectors
from urllib.error import HTTPError
from model import recover_model
from predict import predictor

import json
import os

from flask import Flask
from flask import request
from flask import make_response

# Flask app should start in global layout
app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    global corpus
    global sentence
    global model
    global wv_model

    req = request.get_json(silent=True, force=True)
    intent = req['queryResult']['intent']['displayName']
    textResponse = req['queryResult']['fulfillmentText']
    if intent == 'CorpusSelect':
        cur_corpus = req['queryResult']['parameters']['corpus']
        if cur_corpus != corpus:
            corpus = cur_corpus
    elif intent == 'SentenceInput':
        sentence = req['queryResult']['parameters']['any']
        prediction = predictor([sentence], model[corpus], wv_model)[0]
        textResponse = "The sentence was processed by my Brain :) And it turns out... Wait for it (drumroll), it is %s" % prediction
    elif intent == 'ExplainOutput':
        pass
#         TODO: Explanation stuff
    else:
        print("DEBUG:", req)
    
    res_dict = {"fulfillmentText": textResponse, "source": "webhook"}
    res_json = json.dumps(res_dict, indent=4)
    res = make_response(res_json)
    res.headers['Content-Type'] = 'application/json'
    return res


if __name__ == '__main__':
    global wv_model
    filename = 'GoogleNews-vectors-negative300.bin'
    wv_model = KeyedVectors.load_word2vec_format(filename, binary=True)
    # wv_model = None
    print("WVEC Loaded")
    
    global model
    model = {}
    model['turks'], _, _ = recover_model('nn', 'turks', wv_model)
    
    global corpus
    corpus = None
    
    global sentence
    sentence = None
    
    print('Recovered Model')
    
    port = int(os.getenv('PORT', 5000))
    print("Starting app on port %d" % port)
    app.run(port=port, host='0.0.0.0')
