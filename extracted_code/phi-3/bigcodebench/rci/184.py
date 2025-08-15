import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

# Constants
STOPWORDS = set([...])  # Set of stopwords as defined earlier.

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = ''.join([word for word in text.split() if word not in STOPWORDS])
    return text

def vectorize_text(dataframe, text_column):
    dataframe[text_column] = dataframe[text_column].astype(str).apply(preprocess_text)
    vectorizer = CountVectorizer(min_df=1)  # To avoid adding too many features with rare words.
    X = vectorizer.fit_transform(dataframe[text_column])
    word_count_matrix = X.toarray()
    word_counts = pd.DataFrame(word_count_matrix, columns=vectorizer.get_feature_names_out())
    return word_counts