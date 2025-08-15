import re
import nltk
from gensim.models import Word2Vec

nltk.download('stopwords')

ALPHANUMERIC = re.compile('[\W_]+')

def task_func(texts, stopwords=None):
    if stopwords is None:
        stopwords = set(nltk.corpus.stopwords.words('english'))
    
    cleaned_texts = []
    for text in texts:
        cleaned_text = ALPHANUMERIC.sub(' ', text).lower()
        tokens = [word for word in cleaned_text.split() if word not in stopwords]
        cleaned_texts.append(tokens)
    
    model = Word2Vec(sentences=cleaned_texts, vector_size=100, window=5, min_count=1, workers=4)
    return model