import re

from wordcloud import WordCloud

import matplotlib.pyplot as plt

from collections import Counter


def extract_words(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return re.findall(r'\b\w+\b', text.lower())


def validate_words(words):
    if not words:
        raise ValueError("No words available for word cloud generation.")
    return words


def generate_word_frequencies(words):
    return Counter(words)


def plot_word_cloud(word_freq):
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(word_freq)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    return wordcloud


def task_func(text):
    words = extract_words(text)
    valid_words = validate_words(words)
    word_freq = generate_word_frequencies(valid_words)
    wordcloud = plot_word_cloud(word_freq)
    return wordcloud


# Example usage

# Assuming the function is called within an environment that supports plotting

task_func('Visit https://www.python.org for more info. Python is great. I love Python.')

task_func('Check out this link: http://www.example.com. Machine learning is fascinating.')