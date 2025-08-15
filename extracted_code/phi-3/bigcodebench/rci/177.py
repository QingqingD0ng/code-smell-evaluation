import re
import nltk
from string import punctuation
from collections import Counter
import pandas as pd

def task_func(df, keywords=['like', 'what']):
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    required_columns = ['Title', 'Content']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing_cols)}")

    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, keywords)) + r')\b', re.IGNORECASE)
    filtered_df = df[df['Title'].str.contains(pattern, regex=True)]

    word_freq = Counter()

    for content in filtered_df['Content']:
        text = re.sub(f'[{re.escape(punctuation)}]', '', content).lower()
        words = nltk.word_tokenize(text)
        word_freq.update(words)

    return dict(word_freq)

# Example usage:
# data = {'Title': ['What is happening', 'Nothing special'], 'Content': ['Like what you see? Just normal text.']}
# df = pd.DataFrame(data)
# print(task_func(df))