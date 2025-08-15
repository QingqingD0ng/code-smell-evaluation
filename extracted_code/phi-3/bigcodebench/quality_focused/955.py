import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter

def task_func(mystrings, text):
    if not text:
        raise ValueError("Input text must not be empty")

    text = text.lower()
    modified_strings = [re.sub(r'\s+', '_', s).lower() for s in mystrings]
    text = re.sub(r'\b(' + '|'.join(modified_strings) + r')\b', '_', text)
    text = text.split()

    word_counts = Counter(text)
    words, frequencies = zip(*word_counts.most_common())

    ax = plt.figure().gca()
    ax.bar(words, frequencies)
    plt.xticks(rotation='vertical')
    plt.show()

    return ax