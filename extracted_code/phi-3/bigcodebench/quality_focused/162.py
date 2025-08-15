import re

import matplotlib.pyplot as plt

import numpy as np


def task_func(text, rwidth=0.8):
    words = re.findall(r'\b\w+\b', text)
    lengths = [len(word) for word in words]
    plt.hist(lengths, bins=np.arange(min(lengths), max(lengths) + 1, 1), rwidth=rwidth)
    plt.xlabel('Word Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Word Lengths in Text')
    plt.tight_layout()
    plt.show()
    return plt.gca()


# Example usage:

# ax = task_func('Hello world, this is a test sentence.')