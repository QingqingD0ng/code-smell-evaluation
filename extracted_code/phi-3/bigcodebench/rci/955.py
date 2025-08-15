import numpy as np

import matplotlib.pyplot as plt

import re

from collections import Counter

from typing import List


def format_words(words: List[str]) -> List[str]:

    return [word.replace(" ", "_").lower() for word in words]


def generate_pattern(formatted_words: List[str]) -> str:

    escaped_words = map(re.escape, formatted_words)

    pattern = r'\b(' + '|'.join(escaped_words) + r')\b'

    return pattern


def task_func(mystrings: List[str], text: str) -> plt.Axes:

    if not isinstance(text, str):

        raise TypeError("Input text must be a string.")

    if not text:

        raise ValueError("Input text must not be empty.")


    formatted_words = format_words(mystrings)

    pattern = generate_pattern(formatted_words)

    modified_text = re.sub(pattern, "_", text, flags=re.IGNORECASE)


    words = re.findall(r'\w+', modified_text)

    word_counts = Counter(words)


    ax = plt.figure().gca()

    words, frequencies = zip(*word_counts.items())

    ax.bar(words, frequencies)

    ax.set_xlabel('Words')

    ax.set_ylabel('Frequency')

    plt.show()


    return ax