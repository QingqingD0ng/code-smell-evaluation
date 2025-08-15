import re
import nltk
import string
from collections import Counter

def task_func(df):
    if df.empty or not {'Title', 'Content'}.issubset(df.columns):
        raise ValueError("DataFrame is empty or missing required columns.")

    filtered_df = df[df['Title'].str.lower().str.contains('like|what')]
    texts = filtered_df['Content'].str.lower().str.cat(sep=' ')
    words = nltk.word_tokenize(re.sub(r'['+string.punctuation+']', '', texts))
    word_freq = Counter(word for word in words if word.isalpha())

    return dict(word_freq)