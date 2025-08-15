import re
import matplotlib.pyplot as plt
import numpy as np

def task_func(text, rwidth=0.8):
    words = re.findall(r'\b\w+\b', text)
    word_lengths = [len(word) for word in words]
    if word_lengths:
        plt.hist(word_lengths, bins=range(min(word_lengths), max(word_lengths) + 2), rwidth=rwidth)
        plt.xlabel('Word Length')
        plt.ylabel('Frequency')
        plt.title('Distribution of Word Lengths')
        return plt.gca()
    else:
        return None