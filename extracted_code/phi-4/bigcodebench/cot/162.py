import re
import matplotlib.pyplot as plt
import numpy as np

def task_func(text, rwidth=0.8):
    word_lengths = [len(word) for word in re.findall(r'\b\w+\b', text)]
    if not word_lengths:
        return None

    plt.figure(figsize=(10, 6))
    ax = plt.axes()
    ax.hist(word_lengths, bins=range(1, max(word_lengths) + 2), rwidth=rwidth, edgecolor='black')
    ax.set_xlabel('Word Length')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Word Lengths')
    plt.grid(True)
    return ax