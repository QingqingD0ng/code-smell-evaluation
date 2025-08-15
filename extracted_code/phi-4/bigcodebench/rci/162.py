import re
import matplotlib.pyplot as plt
import numpy as np

def task_func(text, rwidth=0.8):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    
    if not (0 < rwidth <= 1):
        raise ValueError("rwidth must be a float between 0 and 1.")

    words = re.findall(r'\b\w+\b', text)
    word_lengths = [len(word) for word in words]

    if not word_lengths:
        fig, ax = plt.subplots()
        return ax

    fig, ax = plt.subplots()
    ax.hist(word_lengths, bins=range(min(word_lengths), max(word_lengths) + 2), rwidth=rwidth)
    return ax