from flask import Flask, request, jsonify
from pathlib import Path
# import pickle
import pyspark
import numpy as np
from inverted_index_gcp import *
from gensim.models import KeyedVectors
global model
import pickle5 as pickle
from collections import Counter

doc_title_dict = open(f'{"doc_title_dict.pickle"}', 'rb')
doc_title_dict = pickle.load(doc_title_dict)

doc_norm_dict = open(f'{"doc_norm.pickle"}', 'rb')
doc_norm_dict = pickle.load(doc_norm_dict)
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import Counter, OrderedDict
from numpy import linalg as LA
import re

model100 = KeyedVectors.load("word2vec.model100", mmap='r')
model = KeyedVectors.load("word2vec.model", mmap='r')
# find regex pattern s

def get_html_pattern():
    pattern = "<(\"[^\"]\"|'[^']|[^'\">])*>"
    return pattern


def get_time_pattern():
    pattern = "((?:[01][0-2]|2[0-4])(?:\.)?(?:[0-4][0-9])|(?:[0-1]?[0-9]|2[0-5]):(?:([0-5][0-9])):(?:([0-9][0-9]))?$)((AM|am|a\.m\.|PM|p\.m\.)?$|([AP][M]|[ap]\.[m]\.))"
    return pattern


def get_number_pattern():
    pattern = "(?<![\w\+\-,\.])[\+\-]?\d{1,3}((,\d{3})|\d)(\.\d+)?(?!\S?[\w\+\-])"
    return pattern


def get_percent_pattern():
    pattern = "(?<![\w\+\-,\.])[\+\-]?\d{1,3}((,\d{3})|\d)(\.\d+)?%(?!\S?[\w\+\-])"
    return pattern


def get_date_pattern():
    pattern = "((([12][0-9]|(30)|[1-9])\ )?(Apr(il?)?|Jun(e?)?|Sep(tember?)?|Nov(ember?)?)(\ ([12][0-9],|(30,)|[1-9],))?((\ \d\d\d\d)))|((Jan(uary?)?|Mar(ch?)?|May?|Jul(y?)?|Aug(ust?)?|Oct(ober?)?|Dec(ember?)?)(\ ([12][0-9],|3[10],|[1-9],))?((\ \d\d\d\d)))|((([1][0-9]|2[0-8]|[0-9])\ )?(Feb(ruary?)?)(\ ([1][0-9],|2[0-8],|[0-9],))?((\ \d\d\d\d)))"
    return pattern


def get_word_pattern():
    pattern = "(\w+(?:-\w+)+)|(?<!-)(\w+'?\w*)"
    return pattern


RE_TOKENIZE = re.compile(rf"""
(
    # parsing html tags
     (?P<HTMLTAG>{get_html_pattern()})                                  
    # dates
    |(?P<DATE>{get_date_pattern()})
    # time
    |(?P<TIME>{get_time_pattern()})
    # Percents
    |(?P<PERCENT>{get_percent_pattern()})
    # Numbers
    |(?P<NUMBER>{get_number_pattern()})
    # Words
    |(?P<WORD>{get_word_pattern()})
    # space
    |(?P<SPACE>[\s\t\n]+) 
    # everything else
    |(?P<OTHER>.)
)
""",
                         re.MULTILINE | re.IGNORECASE | re.VERBOSE | re.UNICODE)


def filter_text(text):
    filtered = [v for match in RE_TOKENIZE.finditer(text)
                for k, v in match.groupdict().items()
                if v is not None and k not in ['HTMLTAG', 'DATE', 'TIME', 'PERCENT', 'NUMBER', 'SPACE', 'OTHER']]
    return filtered


english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

all_stopwords = english_stopwords.union(corpus_stopwords)





TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
from contextlib import closing


def read_posting_list(inverted, w):
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list




class score:

    def __int__(self):
        pass


    def tokenize(self, text):
        """
        This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

        Parameters:
        -----------
        text: string , represting the text to tokenize.

        Returns:
        -----------
        list of tokens (e.g., list of tokens).
        """

        tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
        ls_lower = filter_text(' '.join(tokens))
        list_of_tokens = [token for token in ls_lower if token not in all_stopwords]
        return list_of_tokens


    def get_candidate_documents_and_scores(self, query_to_search, index):
        """w
        Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
        and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
        Then it will populate the dictionary 'candidates.'
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the document.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                      Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        words,pls: generator for working with posting.
        Returns:
        -----------
        dictionary of candidates. In the following format:
                                                                key: pair (doc_id,term)
                                                                value: tf/tfidf/bm25 score.
        """

        # strings = list(map(lambda x: x[0], model.most_similar(query_vector, topn=3)))
        # query_vector += strings
        # more_tokens = []
        words_in_index = index.df.keys()
        candidates = {}

        for term in np.unique(query_to_search):
            if term in words_in_index:
                posting = read_posting_list(index, term)#pls[words.index(term)]

                for doc_id, score in posting:
                    candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + score


        return candidates


    def retrival(self, section, query_to_search, ranking_method, index):
        """w
        retrival doc_id and title for query_to_search

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                      Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        ranking_method: bm25, binary, cosine similarity
        Returns:
        -----------
        dictionary of candidates. In the following format:
                                                                key: pair (doc_id,term)
                                                                value: tf/tfidf/bm25 score.
        """
        #change - query to search according to the section wanted
        if section == "body":
            pass
        elif section == "title":
            query_to_search = [token + '_t' for token in query_to_search]
        elif section == "anchor":
            query_to_search = [token + '_a' for token in query_to_search]
        elif section == "all":
            words_in_index = index.df.keys()
            more_tokens = []

            #use query eapansion, if query len between 1,3 use 100 dimesions, if its above or eaual to 3, 200 dimensions.

            if 3>len(query_to_search)>1:
                more_tokens = list(map(lambda x: x[0], model.most_similar(query_to_search, topn=3)))
            if len(query_to_search)>=3:
                more_tokens = list(map(lambda x: x[0], model100.most_similar(query_to_search, topn=1)))


            query_to_search_t = [token + '_t' for token in query_to_search]
            query_to_search_a = [token + '_a' for token in query_to_search]
            query_to_search += query_to_search_t + query_to_search_a +more_tokens
            query_to_search = sorted(query_to_search)

        candidates = self.get_candidate_documents_and_scores(query_to_search, index)


        #calculate the order of the retrival documents and return
        if ranking_method == "binary":
            docs = []
            for (doc_id, token), _ in candidates.items():
                if doc_id in doc_title_dict: key = (doc_id, doc_title_dict[doc_id])
                else: key = (doc_id, str(doc_id))
                docs += [key]
            countered = Counter(docs)
            res = sorted(countered, key=countered.get, reverse=True)

        elif ranking_method == "cosine similarity":
            cosine_docs_dict = defaultdict(float)
            query_counter = Counter(query_to_search)
            for (doc_id, token), tfidf in candidates.items():
                if doc_id in doc_title_dict: key = (doc_id, doc_title_dict[doc_id])
                else: key = (doc_id, str(doc_id))
                cosine_docs_dict[key] += (tfidf * query_counter[token]) / doc_norm_dict[doc_id] #query norm is the same for all calculations so its un nedeeded
            res = sorted(cosine_docs_dict, key=cosine_docs_dict.get, reverse=True)

        elif ranking_method == "bm25":

            bm25_dict = defaultdict(float)
            for (doc_id, token), score in candidates.items():
                if doc_id in doc_title_dict: key = (doc_id, doc_title_dict[doc_id])
                else: key = (doc_id, str(doc_id))
                if token.endswith('_a'):
                    score*=1.25
                elif token.endswith('_t'):
                    score*=0.45
                else:
                    score = 0.9*score
                # dict_key = (doc_id, doc_title_dict[doc_id])
                bm25_dict[key] += score
            res = sorted(bm25_dict, key=bm25_dict.get, reverse=True)

        return res
