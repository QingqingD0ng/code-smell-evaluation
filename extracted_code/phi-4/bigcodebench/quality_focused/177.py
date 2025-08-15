import re
import nltk
from string import punctuation
import pandas as pd
from collections import Counter

def task_func(df):
    if df.empty or not {'Title', 'Content'}.issubset(df.columns):
        raise ValueError("DataFrame is empty or missing 'Title' or 'Content' columns.")
    
    keyword_pattern = re.compile(r'\b(?:like|what)\b', re.IGNORECASE)
    filtered_df = df[df['Title'].str.contains(keyword_pattern, flags=re.IGNORECASE)]
    
    content_texts =''.join(filtered_df['Content'].tolist())
    words = nltk.word_tokenize(content_texts)
    filtered_words = [word.lower() for word in words if word.lower() not in punctuation]
    
    return dict(Counter(filtered_words))

# Example usage:
# data = {'Title': ['What is happening', 'Nothing special'], 'Content': ['Like what you see?', 'Just normal text.']}
# df = pd.DataFrame(data)
# print(task_func(df))