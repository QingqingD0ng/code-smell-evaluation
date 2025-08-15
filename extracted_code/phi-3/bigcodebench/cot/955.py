import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter

def task_func(mystrings, text):
    if not text:
        raise ValueError("Input text must not be empty.")
    
    modified_strings = [s.replace(" ", "_") for s in mystrings]
    text = text.lower()
    words = re.findall(r'\b(?:' + '|'.join(map(re.escape, modified_strings)) + r')\b', text)
    word_counts = Counter(words)
    sorted_words = sorted(word_counts.keys(), key=lambda word: text.index(word))
    frequencies = [word_counts[word] for word in sorted_words]
    ax = plt.figure().gca()
    ax.bar(sorted_words, frequencies)
    plt.show()
    return ax