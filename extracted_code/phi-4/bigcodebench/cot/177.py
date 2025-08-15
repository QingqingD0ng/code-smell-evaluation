import re
import nltk
from string import punctuation
import pandas as pd

def task_func(df):
    if df.empty:
        raise ValueError("DataFrame is empty")
    if 'Title' not in df.columns or 'Content' not in df.columns:
        raise ValueError("DataFrame must contain 'Title' and 'Content' columns")

    keywords = ["like", "what"]
    pattern = r'\b(?:' + '|'.join(keywords) + r')\b'
    filtered_df = df[df['Title'].str.lower().str.contains(pattern, case=False)]
    
    word_freq = {}

    for content in filtered_df['Content']:
        content = re.sub(f"[{re.escape(punctuation)}]", "", content)
        words = nltk.word_tokenize(content)
        for word in words:
            word = word.lower()
            if word:
                word_freq[word] = word_freq.get(word, 0) + 1

    return word_freq