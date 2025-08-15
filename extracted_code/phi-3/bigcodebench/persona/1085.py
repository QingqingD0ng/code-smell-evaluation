import re

from collections import Counter

import matplotlib.pyplot as plt


def task_func(text):
    text = text.lower()
    words = re.findall(r'\b[a-z]+\b', text)
    word_counts = Counter(words)
    common_words = word_counts.most_common(10)
    fig, ax = plt.subplots()
    ax.bar([word[0] for word in common_words], [word[1] for word in common_words])
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title('Top 10 Most Common Words')
    plt.xticks(rotation=45)
    return common_words, ax