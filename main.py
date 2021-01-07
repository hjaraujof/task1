from helpers import BasicTextExtractFromWebsite, SingleDocTFIDFHelper
from flask import Flask, jsonify
from flask import request
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/tfidf')
def tfidf_url():
    url = request.args.get('url', '')
    limit = request.args.get('limit', '')
    if url != '' and limit != '':
        scraped_text = BasicTextExtractFromWebsite(url).scrapeTHSOOT()
        tfidf = SingleDocTFIDFHelper(scraped_text)
        tfidf.normalize()
        tfidf.execTFIDF(limit)
        return jsonify({
            "tfidf": tfidf.response_obj,
            "word_count": tfidf.total_words,
            "word_count_dict": tfidf.word_count_dict
        })
    else:
        return jsonify(error='Error on params')
