import re
import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Constants
ALPHANUMERIC = re.compile('[\W_]+')


def task_func(texts, stopwords_list=None):
    # Use nltk's stopwords if none are provided
    if stopwords_list is None:
        stopwords_list = set(stopwords.words('english'))

    # Preprocess texts
    processed_texts = []
    for text in texts:
        # Remove non-alphanumeric characters, lowercase, and split into words
        words = ALPHANUMERIC.sub(' ', text).lower().split()
        # Remove stopwords
        filtered_words = [word for word in words if word not in stopwords_list]
        processed_texts.append(filtered_words)

    # Train Word2Vec model
    model = Word2Vec(processed_texts, vector_size=100, window=5, min_count=1, workers=4)

    return model