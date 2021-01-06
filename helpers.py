import re
import string
import unidecode
import math
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class SingleDocTFIDFHelper:
    def __init__(self, original_text):
        self.original_text = original_text
        self.stop_words = set(stopwords.words('english'))
        self.punctuation_table = string.maketrans('', '', string.punctuation)
        self.word_set = []
        self.word_count_dict = {}
        self.normalized_words = []
        self.total_words = 0
        self.tf_dict = {}
        self.idf_dict = {}
        self.tfidf_dict = {}
        self.response_obj = {}
        self.response_obj['terms'] = []

    # Remove single/repeated whitespaces from input text
    def remove_whitespaces(str_input):
        return re.split(' +', str_input)

    # Tokenizes the sentences within given string
    def sentence_tokenizer(str_input):
        return sent_tokenize(str_input)

    # Tokenizes the words within given string
    def word_tokenizer(str_input):
        return word_tokenize(str_input)

    # Removes all stopwords from string
    def remove_stopwords(self, word_array):
        filtered_words = []
        for word in word_array:
            if word not in self.stop_words:
                filtered_words.append(word)
        return filtered_words

    # Lowercases string
    def to_lower_case(str_input):
        return str_input.lower()

    # Removes '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' punctuation marks
    def remove_punctuation(self, str_input):
        return str_input.translate(self.punctuation_table)

    # Removes remaining special unicode chars, like accents or quotation marks
    # Basically converts a unicode string to an ASCII string
    def remove_unicode_chars(str_input):
        return unidecode.unidecode(str_input)

    # Normalize the original text and return the relevant terms
    def normalize(self):
        text = self.remove_whitespaces(self.original_text)
        text = self.remove_unicode_chars(text)
        text = self.to_lower_case(text)
        sentences = self.sentence_tokenizer(text)
        normalized_words = []
        for sent in sentences:
            sent = self.remove_punctuation(sent)
            sent = sent.strip()  # strip leading/trailing whitespace
            words = self.word_tokenizer(sent)
            filtered_words = self.remove_stopwords(words)
            normalized_words.extend(filtered_words)
        self.normalized_words = normalized_words
        self.word_set = set(normalized_words)
        self.word_count_dict = dict.fromkeys(self.word_set, 0)
        self.total_words = len(normalized_words)

    # Count frecuency of each relevant term
    def count_words(self):
        for word in self.normalized_words:
            self.word_count_dict[word] += 1

    # Calculate the term frecuency
    def calculate_tf(self):
        for word, f in self.word_count_dict.items():
            self.tf_dict[word] = f/float(self.total_words)

    # Calculate the inverse data frecuency
    def calculate_idf(self):
        N = 1  # We only use a single document
        self.idf_dict = dict.fromkeys(self.word_set, 0)
        for word, val in self.word_count_dict.items():
            if val > 0:
                self.idf_dict[word] += 1
        for word, val in self.idf_dict.items():
            self.idf_dict[word] = math.log10(N / float(val))

    # Finally calculate TFIDF
    def calculate_tfidf(self):
        for word, tf in self.tf_dict.items():
            self.tfidf_dict[word] = tf * self.idf_dict[word]

    # Execute all
    def exec(self, limit):
        self.normalize()
        self.count_words()
        self.calculate_tf()
        self.calculate_idf()
        self.calculate_tfidf()
        self.convert_dict_to_response_obj(limit)
        return self.response_obj

    # Sort desc and limit final TFIDF dict
    def sort_n_limit_dic(d, limit):
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
        final_tfidf_dict = self.sort_n_limit_dic(self.tfidf_dict, limit)
        for term, val in final_tfidf_dict.items():
            self.response_obj['terms'].append({'term': term, 'tf-idf': val})
