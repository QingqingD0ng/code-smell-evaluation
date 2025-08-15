import re
import nltk
from gensim.models import Word2Vec

ALPHANUMERIC = re.compile('[\W_]+')

def task_func(texts, stopwords=None):
    if stopwords is None:
        stopwords = set(nltk.corpus.stopwords.words('english'))
    
    processed_texts = [
        [word for word in ALPHANUMERIC.sub(' ', text).lower().split() if word and word not in stopwords]
        for text in texts
    ]
    
    return Word2Vec(sentences=processed_texts, vector_size=100, window=5, min_count=1, workers=4)