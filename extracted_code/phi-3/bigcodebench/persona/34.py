import re

from wordcloud import WordCloud

import matplotlib.pyplot as plt


def task_func(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'https\S+', '', text)
    words = re.findall(r'\w+', text)
    if not words:
        raise ValueError('No words available to generate a word cloud after removing URLs.')
    wordcloud = WordCloud(width=800, height=400).generate(' '.join(words))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    return wordcloud


# Example usage:

# print(task_func('Visit https://www.python.org for more info. Python is great. I love Python.'))

# print(task_func('Check out this link: http://www.example.com. Machine learning is fascinating.'))