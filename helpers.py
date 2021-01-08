import re
import unidecode
import math
import urllib.request
import itertools
import csv
import sys
import time
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
csv.field_size_limit(sys.maxsize)


class SingleDocTFIDFHelper:
    def __init__(self, original_text):
        tic = time.time()
        self.original_text = original_text
        self.word_set = []
        self.word_count_dict = {}
        self.normalized_words = []
        self.total_words = 0
        self.tf_dict = {}
        self.idf_dict = {}
        self.tfidf_dict = {}
        self.response_obj = {}
        self.response_obj['terms'] = []
        nltk.download('stopwords')
        nltk.download('punkt')
        from nltk.corpus import stopwords
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        toc = time.time()
        print('__init__ took ' + str(toc - tic) + 's')

    # Remove single/repeated whitespaces from input text
    def remove_whitespaces(self, str_input):
        return re.split(' +', str_input)

    # Remove all line breaks from input text
    def remove_linebreaks_whitespaces(self, str_input):
        text = " ".join(str_input.splitlines())
        return " ".join(text.split())

    # Tokenizes the sentences within given string
    def sentence_tokenizer(self, str_input):
        return sent_tokenize(str_input)

    # Tokenizes the words within given string
    def word_tokenizer(self, str_input):
        return word_tokenize(str_input)

    # Removes all stopwords from string
    def remove_stopwords(self, word_array):
        filtered_words = []
        for word in word_array:
            if ((word not in self.stop_words) and
                    (len(word) > 1) and (bool(re.match("^[a-z]*$", word)))):
                word = self.remove_whitespaces(word)
                filtered_words.append(word)
        return filtered_words

    # Lowercases string
    def to_lower_case(self, str_input):
        return str_input.lower()

    # Removes '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' punctuation marks
    def remove_punctuation(self, str_input):
        # return str_input.translate(string.punctuation)
        return re.sub(r'[^\w\s]', '', str_input)

    # Removes remaining special unicode chars, like accents or quotation marks
    # Basically converts a unicode string to an ASCII string
    def remove_unicode_chars(self, str_input):
        return unidecode.unidecode(str_input)

    # Normalize the original text and return the relevant terms
    def normalize(self):
        # text = self.remove_punctuation(self.original_text)
        # print(text)
        tic = time.time()
        text = self.remove_unicode_chars(self.original_text)
        text = self.remove_linebreaks_whitespaces(text)
        text = self.remove_punctuation(text)
        # print(text)
        text = self.to_lower_case(text)
        normalized_words = []
        words = self.word_tokenizer(text)
        filtered_words = self.remove_stopwords(words)
        normalized_words.extend(filtered_words)
        flattened = list(itertools.chain(*normalized_words))
        self.normalized_words = flattened
        # print(self.normalized_words)
        self.word_set = set(self.normalized_words)
        self.idf_dict = dict.fromkeys(self.word_set, 0)
        self.word_count_dict = dict.fromkeys(self.word_set, 0)
        self.total_words = len(self.normalized_words)
        toc = time.time()
        print('normalize: ' + str(toc - tic) + 's')

    # Count frecuency of each relevant term
    def count_words(self):
        tic = time.time()
        for word in self.normalized_words:
            self.word_count_dict[word] += 1
        toc = time.time()
        print('count_words: ' + str(toc - tic) + 's')

    # Calculate the term frecuency
    def calculate_tf(self):
        tic = time.time()
        for word, f in self.word_count_dict.items():
            self.tf_dict[word] = f/float(self.total_words)
        toc = time.time()
        print('calculate_tf: ' + str(toc - tic) + 's')

    # Calculate the inverse data frecuency
    def calculate_idf(self):
        tic = time.time()
        with open('articles1.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            read_toc = time.time()
            print('csvReader took ' + str(read_toc - tic) + 's')
            N = 0
            for row in reader:
                N += 1
                for word, value in self.word_count_dict.items():
                    if value > 1:
                        content = row['content']
                        if f' {word} ' in f' {content} ':
                            self.idf_dict[word] += 1
            count_toc = time.time()
            print('Initial idf calc took ' + str(count_toc - read_toc) + 's')
        for word, val in self.idf_dict.items():
            if val > 0:
                self.idf_dict[word] = math.log10(N / float(val))
        toc = time.time()
        print('calculate_idf: ' + str(toc - count_toc) + 's')

    # Finally calculate TFIDF
    def calculate_tfidf(self):
        for word, tf in self.tf_dict.items():
            self.tfidf_dict[word] = tf * self.idf_dict[word]

    # Execute all
    def execTFIDF(self, limit):
        self.normalize()
        self.count_words()
        self.calculate_tf()
        self.calculate_idf()
        self.calculate_tfidf()
        self.convert_dict_to_response_obj(limit)
        return self.response_obj

    # Sort desc and limit final TFIDF dict
    def sort_n_limit_dic(self, d, limit):
        result = {}
        for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True):
            if(limit > 0):
                result[k] = v
                limit -= 1
            if(limit == 0):
                break
        return result

    # Convert tfidf dic to desired format
    def convert_dict_to_response_obj(self, limit):
        final_tfidf_dict = self.sort_n_limit_dic(self.tfidf_dict, int(limit))
        for term, val in final_tfidf_dict.items():
            self.response_obj['terms'].append({'term': term, 'tf-idf': val})


class BasicTextExtractFromWebsite:
    def __init__(self, url):
        self.site_url = url

    def scrapeTHSOOT(self):
        content = urllib.request.urlopen(self.site_url)
        read_content = content.read()
        soup = BeautifulSoup(read_content, 'html.parser')
        return soup.get_text(" ")
