import re
import nltk
from string import punctuation
from collections import Counter

def task_func(df):
    if df.empty or 'Title' not in df.columns or 'Content' not in df.columns:
        raise ValueError("DataFrame is empty or missing required columns.")

    # Initialize regex pattern for case-insensitive matching of "like" or "what"
    pattern = r'\blike\b|\bwhat\b'
    
    # Filter articles based on title keywords
    filtered_df = df[df['Title'].str.contains(pattern, case=False, na=False)]
    
    # Initialize a Counter object to track word frequencies
    word_freq = Counter()
    
    # Process each article's content
    for content in filtered_df['Content']:
        # Remove punctuation and convert to lowercase
        text = re.sub(f'[{re.escape(punctuation)}]', '', content).lower()
        # Tokenize the text into words
        words = nltk.word_tokenize(text)
        # Update the word frequency count
        word_freq.update(words)
    
    # Return the word frequency dictionary
    return dict(word_freq)

# Example usage:
# import pandas as pd
# data = {'Title': ['What is happening', 'Nothing special'], 'Content': ['Like what you see?', 'Just normal text.']}
# df = pd.DataFrame(data)
# print(task_func(df))