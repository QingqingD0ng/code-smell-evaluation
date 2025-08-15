import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

STOPWORDS = ['i','me','my','myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself','she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once']

def task_func(dataframe, text_column):
    vectorizer = CountVectorizer(stop_words=STOPWORDS, lowercase=True, token_pattern=r'\b[a-zA-Z]{2,}\b')
    text_data = dataframe[text_column].astype(str)
    X = vectorizer.fit_transform(text_data)
    word_count_matrix = X.toarray()
    word_counts = pd.DataFrame(word_count_matrix, columns=vectorizer.get_feature_names_out())
    return word_counts