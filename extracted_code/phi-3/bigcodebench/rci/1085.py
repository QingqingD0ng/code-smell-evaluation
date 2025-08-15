import re

from collections import Counter

import matplotlib.pyplot as plt


def process_text(text):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    return re.sub(r'[^\w\s]', '', text.lower()).split()


def plot_word_frequencies(word_counts):
    if not isinstance(word_counts, Counter):
        raise ValueError("Input must be a Counter object containing word frequencies.")
    
    common_words = word_counts.most_common(10)
    words, counts = zip(*common_words)
    
    fig, ax = plt.subplots()
    ax.bar(words, counts)
    ax.set_xticklabels(words)
    ax.set_xlabel('Words')
    ax.set_ylabel('Counts')
    ax.set_title('Top 10 Most Common Words')
    
    return ax


# Example usage:

try:
    text = "This is a sample text. This text contains sample words like 'text','sample', and 'words'."
    words = process_text(text)
    ax = plot_word_frequencies(Counter(words))
    plt.show()
except ValueError as e:
    print(f"Error: {e}")