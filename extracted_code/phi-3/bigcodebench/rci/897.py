import numpy as np
import matplotlib.pyplot as plt
import random

NUMBERS = list(range(1, 7))

def task_func(rolls, seed=None):
    if seed is not None:
        random.seed(seed)
    results = [random.choice(NUMBERS) for _ in range(rolls)]
    unique, counts = np.unique(results, return_counts=True)
    frequencies = counts / rolls
    fig, ax = plt.subplots()
    ax.bar(unique, frequencies, tick_label=unique, alpha=0.7)
    ax.set_title("Histogram of Dice Rolls")
    ax.set_xlabel("Dice Value")
    ax.set_ylabel("Frequency")
    plt.show()
    return frequencies, ax