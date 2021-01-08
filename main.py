from helpers import BasicTextExtractFromWebsite, SingleDocTFIDFHelper
from flask import Flask, jsonify
from flask import request
import time
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/tfidf')
def tfidf_url():
    url = request.args.get('url', '')
    limit = request.args.get('limit', '')
    if url != '' and limit != '':
        tic = time.time()
        scraped_text = BasicTextExtractFromWebsite(url).scrapeTHSOOT()
        toc = time.time()
        print('scrape content from website took: ' + str(toc - tic) + 's')
        tfidf = SingleDocTFIDFHelper(scraped_text)
        tfidf.execTFIDF(limit)
        toc2 = time.time()
        print('the whole thing took: ' + str(toc2 - tic) + 's')
        return jsonify(tfidf.response_obj)
    else:
        return jsonify(error='Error on params')
