import re
import nltk
from string import punctuation

nltk.download('punkt')

def task_func(df):
    if df.empty or 'Title' not in df.columns or 'Content' not in df.columns:
        raise ValueError("DataFrame is empty or missing 'Title' and/or 'Content' columns.")

    filtered_content = []
    for content in df.loc[df['Title'].str.contains(r'like|what', case=False, na=False), 'Content']:
        words = nltk.word_tokenize(content)
        filtered_content.extend(word for word in words if word.lower() not in punctuation)

    return nltk.FreqDist(filtered_content).most_common()