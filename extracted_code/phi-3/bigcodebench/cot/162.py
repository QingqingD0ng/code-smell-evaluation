import re
import matplotlib.pyplot as plt
import numpy as np

def task_func(text, rwidth=0.8):
    words = re.findall(r'\b\w+\b', text.lower())
    lengths = [len(word) for word in words]
    plt.hist(lengths, rwidth=rwidth)
    plt.xlabel('Word Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Word Lengths')
    plt.show()
    return plt.gca()