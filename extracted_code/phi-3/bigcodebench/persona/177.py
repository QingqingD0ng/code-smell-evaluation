import re

import nltk

from string import punctuation

from collections import Counter


def task_func(df):
    if df.empty or not all(col in df.columns for col in ['Title', 'Content']):
        raise ValueError("DataFrame is empty or missing required columns.")
    
    keywords = ["like", "what"]
    regular_expression = re.compile(r'\b(?:{})\b'.format('|'.join(map(re.escape, keywords))), re.IGNORECASE)
    articles_with_keywords = df[df['Title'].apply(lambda x: bool(regular_expression.search(x)))]
    
    if articles_with_keywords.empty:
        return {}
    
    all_words =''.join(articles_with_keywords['Content']).lower()
    words = nltk.word_tokenize(all_words)
    words = [word for word in words if word not in punctuation]
    
    return dict(Counter(words))