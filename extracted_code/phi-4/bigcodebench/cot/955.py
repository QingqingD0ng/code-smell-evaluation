import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter

def task_func(mystrings, text):
    if not text:
        raise ValueError("Input text must not be empty.")

    for phrase in mystrings:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        text = pattern.sub(phrase.replace(' ', '_'), text)

    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)

    unique_words = []
    seen = set()
    for word in words:
        if word not in seen:
            unique_words.append(word)
            seen.add(word)

    frequencies = [word_counts[word] for word in unique_words]

    fig, ax = plt.subplots()
    ax.bar(unique_words, frequencies)
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title('Word Frequency Plot')
    plt.xticks(rotation=90)
    plt.tight_layout()

    return ax