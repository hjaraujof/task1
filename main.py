import urllib
from flask import Flask
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
        return urllib.parse.unquote(url) + ' ' + limit, 200
    else:
        return 'Error on params', 400
