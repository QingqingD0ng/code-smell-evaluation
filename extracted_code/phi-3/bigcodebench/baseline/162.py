import re
import matplotlib.pyplot as plt
import numpy as np

def task_func(text, rwidth=0.8):
    words = re.findall(r'\b\w+\b', text)
    lengths = [len(word) for word in words]
    plt.hist(lengths, bins=range(min(lengths), max(lengths) + 2), rwidth=rwidth)
    plt.xlabel('Word Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Word Lengths')
    plt.show()
    return plt.gca()