import re

import matplotlib.pyplot as plt


def task_func(text, rwidth=0.8):

    word_lengths = [len(word) for word in re.findall(r'\b\w+\b', text)]

    plt.hist(word_lengths, bins=range(min(word_lengths), max(word_lengths) + 1), rwidth=rwidth)

    plt.xlabel('Word Length')

    plt.ylabel('Frequency')

    plt.title('Distribution of Word Lengths in Text')

    return plt.gca()