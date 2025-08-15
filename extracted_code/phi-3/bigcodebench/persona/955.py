import numpy as np

import matplotlib.pyplot as plt

import re

from collections import Counter


def task_func(mystrings, text):
    if not text:
        raise ValueError("Input text must not be empty")

    modified_words = [s.replace(" ", "_") for s in mystrings]
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    words_counter = Counter(words)
    modified_words_counter = Counter([w.replace(" ", "_") for w in words])

    unique_words = list(modified_words_counter.keys())
    frequencies = list(modified_words_counter.values())

    plt.bar(unique_words, frequencies)
    plt.show()

    return plt.gca()


# Example usage:

ax = task_func(['Lorem ipsum', 'consectetur adipiscing'], 'Lorem ipsum dolor sit amet lorem Ipsum')