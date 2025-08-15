import re

from wordcloud import WordCloud

import matplotlib.pyplot as plt


def task_func(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S*@\S*\s?', '', text)
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        raise ValueError("No words available to generate a word cloud after removing URLs.")
    word_freq = {word: 0 for word in words}
    for word in words:
        word_freq[word] += 1
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(word_freq)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    return wordcloud


# Example usage

# Assuming the function is called within an environment that supports plotting (e.g., Jupyter Notebook)

task_func('Visit https://www.python.org for more info. Python is great. I love Python.')

task_func('Check out this link: http://www.example.com. Machine learning is fascinating.')