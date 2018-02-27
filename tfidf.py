import os
from stopwords import del_stop_words
import numpy as np


def get_all_vector(file_path, historypath, stop_words_set):
    names = [os.path.join(file_path, f) for f in os.listdir(file_path)]
    hisnames = [os.path.join(historypath, f) for f in os.listdir(historypath)]
    names.extend(hisnames)
    posts = [open(name, encoding="utf-8").read() for name in names]
    docs = []
    word_set = set()
    for post in posts:
        doc = del_stop_words(post, stop_words_set)
        docs.append(doc)
        word_set = set(doc)

    word_set = list(word_set)
    docs_vsm = []
    for doc in docs:
        temp_vector = []
        for word in word_set:
            temp_vector.append(doc.count(word) * 1.0)
        docs_vsm.append(temp_vector)

    docs_matrix = np.array(docs_vsm)
    column_sum = [float(len(np.nonzero(docs_matrix[:, i])[0])) for i in range(docs_matrix.shape[1])]
    column_sum = np.array(column_sum)
    column_sum = docs_matrix.shape[0] / column_sum
    idf = np.log(column_sum)
    idf = np.diag(idf)
    for doc_v in docs_matrix:
        if doc_v.sum() == 0:
            doc_v = doc_v / 1
        else:
            doc_v = doc_v / (doc_v.sum())
        tfidf = np.dot(docs_matrix, idf)
        return names, tfidf
