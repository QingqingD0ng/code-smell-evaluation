import re
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords

ALPHANUMERIC = re.compile('[\W_]+')

def task_func(texts, stopwords=None):
    if stopwords is None:
        stopwords = set(stopwords.words('english'))

    cleaned_texts = []
    for text in texts:
        text = ALPHANUMERIC.sub(' ', text).lower()
        words = nltk.word_tokenize(text)
        words = [word for word in words if word not in stopwords]
        cleaned_texts.append(words)

    model = Word2Vec(cleaned_texts, vector_size=100, window=5, min_count=1, workers=4)
    return model