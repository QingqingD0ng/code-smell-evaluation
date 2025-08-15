import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

STOPWORDS = set(['i','me','my','myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                 'yourselves', 'he', 'him', 'his', 'himself','she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
                 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
                 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
                 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                 'further', 'then', 'once'])

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    text =''.join(word for word in text.split() if word not in STOPWORDS)
    return text

def vectorize_text(dataframe):
    vectorizer = CountVectorizer(token_pattern=r'\b[^\d\W]+\b')
    text_data = dataframe[text_column].apply(clean_text)
    X = vectorizer.fit_transform(text_data)
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_)