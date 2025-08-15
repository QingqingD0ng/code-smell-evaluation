import re
import matplotlib.pyplot as plt
import numpy as np

def task_func(text, rwidth=0.8):
    words = re.findall(r'\b\w+\b', text)
    word_lengths = [len(word) for word in words]

    if not word_lengths:
        plt.figure()
        ax = plt.gca()
        plt.show()
        return ax

    plt.figure()
    ax = plt.subplot()
    ax.hist(word_lengths, bins=range(min(word_lengths), max(word_lengths) + 2), rwidth=rwidth)
    plt.show()
    return ax