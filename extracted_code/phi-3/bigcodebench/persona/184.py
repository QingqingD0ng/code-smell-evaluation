import pandas as pd

import re

from sklearn.feature_extraction.text import CountVectorizer


STOPWORDS = [
    'i','me','my','myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
    'his', 'himself','she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
]


def task_func(dataframe, text_column):
    dataframe[text_column] = dataframe[text_column].astype(str)

    # Remove numbers and punctuation
    dataframe[text_column] = dataframe[text_column].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

    # Remove stopwords
    vectorizer = CountVectorizer(stop_words=STOPWORDS)
    word_counts = vectorizer.fit_transform(dataframe[text_column])

    # Convert to DataFrame
    word_count_df = pd.DataFrame(