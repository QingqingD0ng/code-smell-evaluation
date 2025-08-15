import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

STOPWORDS = {'i','me','my','myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
             'yourselves', 'he', 'him', 'his', 'himself','she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
             'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
             'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
             'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
             'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
             'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
             'further', 'then', 'once'}

NUMBERS_PATTERN = r'\d+'
PUNCTUATION_PATTERN = r'[^\w\s]'

def preprocess_text(text):
    text = text.lower()
    text = re.sub(NUMBERS_PATTERN, '', text)
    text = re.sub(PUNCTUATION_PATTERN, '', text)
    return''.join(word for word in text.split() if word not in STOPWORDS)

def task_func(dataframe, text_column):
    if text_column not in dataframe.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")
    
    dataframe['processed_text'] = dataframe[text_column].apply(preprocess_text)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(dataframe['processed_text'])
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())