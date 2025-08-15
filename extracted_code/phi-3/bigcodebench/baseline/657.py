import re

import nltk

from nltk.corpus import stopwords

from gensim.models import Word2Vec

from nltk.tokenize import word_tokenize


def task_func(texts, stopwords=None):

    nltk.download('punkt')

    nltk.download('stopwords')

    if stopwords is None:

        stopwords = set(stopwords.words('english'))


    cleaned_texts = []

    for text in texts:

        text = ALPHANUMERIC.sub(' ', text)

        text = text.lower()

        tokens = word_tokenize(text)

        tokens = [word for word in tokens if word not in stopwords]

        cleaned_texts.append(tokens)


    model = Word2Vec(sentences=cleaned_texts, vector_size=100, window=5, min_count=1, workers=4)

    return model