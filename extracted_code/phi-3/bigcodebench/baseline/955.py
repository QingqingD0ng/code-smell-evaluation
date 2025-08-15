import numpy as np

import matplotlib.pyplot as plt

import re

from collections import Counter


def task_func(mystrings, text):

    if not text:

        raise ValueError("Input text must not be empty.")


    # Replace spaces in given words with underscores and change to lower case

    formatted_words = [word.replace(" ", "_").lower() for word in mystrings]


    # Prepare text by replacing given formatted words with underscores and changing to lower case

    pattern = r'\b(' + '|'.join(map(re.escape, formatted_words)) + r')\b'

    modified_text = re.sub(pattern, "_", text, flags=re.IGNORECASE)


    # Count word frequencies

    words = re.findall(r'\w+', modified_text)

    word_counts = Counter(words)


    # Plotting

    ax = plt.figure().gca()

    words, frequencies = zip(*word_counts.items())

    ax.bar(words, frequencies)

    ax.set_xlabel('Words')

    ax.set_ylabel('Frequency')

    plt.show()


    return ax