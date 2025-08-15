import re
import nltk
from string import punctuation
import pandas as pd

def task_func(df):
    if df.empty or not {'Title', 'Content'}.issubset(df.columns):
        raise ValueError("DataFrame is empty or missing 'Title' and 'Content' columns")

    filtered_df = df[df['Title'].str.lower().str.contains(r'\b(like|what)\b')]

    nltk.download('punkt', quiet=True)
    word_list = []
    for content in filtered_df['Content']:
        words = nltk.word_tokenize(content)
        words = [word.lower() for word in words if word.isalpha()]
        word_list.extend(words)

    frequency = {}
    for word in word_list:
        frequency[word] = frequency.get(word, 0) + 1

    return frequency