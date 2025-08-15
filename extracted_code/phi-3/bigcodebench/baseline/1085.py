import re

from collections import Counter

import matplotlib.pyplot as plt


def task_func(text):
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    word_counts = Counter(words)
    common_words = word_counts.most_common(10)
    
    fig, ax = plt.subplots()
    ax.bar([word for word, count in common_words], [count for word, count in common_words])
    ax.set_xlabel('Words')
    ax.set_ylabel('Counts')
    ax.set_title('Top 10 Most Common Words')
    ax.set_xticks([word for word, count in common_words])
    ax.set_xticklabels([word for word, count in common_words])
    
    return common_words, ax


# Example usage:

common_words, ax = task_func("This is a sample text. This text contains sample words like 'text','sample', and 'words'.")

print(common_words)

plt.show()