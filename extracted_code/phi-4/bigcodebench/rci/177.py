import re
import nltk
from string import punctuation
import pandas as pd
from collections import Counter

def task_func(df, keywords=None, title_col='Title', content_col='Content'):
    if df.empty or not {title_col, content_col}.issubset(df.columns):
        raise ValueError(f"DataFrame is empty or missing '{title_col}' and '{content_col}' columns")

    if keywords is None:
        keywords = ['like', 'what']

    keyword_pattern = r'\b(' + '|'.join(map(re.escape, keywords)) + r')\b'
    filtered_df = df[df[title_col].str.lower().str.contains(keyword_pattern, regex=True)]

    nltk.download('punkt', quiet=True)
    word_list = []
    for content in filtered_df[content_col]:
        words = nltk.word_tokenize(content)
        words = [word.lower() for word in words if word.isalpha()]
        word_list.extend(words)

    frequency = Counter(word_list)

    return dict(frequency)