import re
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords

ALPHANUMERIC = re.compile('[\W_]+')

def clean_text(texts, stop_words):
    cleaned_texts = []
    for text in texts:
        text = ALPHANUMERIC.sub(' ', text)
        text = text.lower()
        words = nltk.word_tokenize(text)
        words = [word for word in words if word not in stop_words]
        cleaned_texts.append(' '.join(words))
    return cleaned_texts

def task_func(texts, stopwords=None):
    if stopwords is None:
        stopwords = set(stopwords.words('english'))
    cleaned_texts = clean_text(texts, stopwords)
    model = Word2Vec(cleaned_texts, vector_size=100, window=5, min_count=1, workers=4)
    return model

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('stopwords')
    texts = ["Hello, World!", "Machine Learning is great", "Python is my favorite programming language"]
    model = task_func(texts)
    print(model.wv['python'])