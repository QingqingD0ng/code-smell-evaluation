import re

from collections import Counter

import matplotlib.pyplot as plt


def task_func(text):
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    counter = Counter(words)
    common_words = counter.most_common(10)
    fig, ax = plt.subplots()
    ax.bar([word for word, count in common_words], [count for word, count in common_words])
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title('Top 10 Most Common Words')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    return common_words


# Example usage

common_words = task_func("This is a sample text. This text contains sample words like 'text','sample', and 'words'.")

print(common_words)