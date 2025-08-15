import numpy as np
import matplotlib.pyplot as plt
import random

NUMBERS = list(range(1, 7))

def task_func(rolls, seed=None):
    if seed is not None:
        random.seed(seed)
    results = [random.choice(NUMBERS) for _ in range(rolls)]
    unique, counts = np.unique(results, return_counts=True)
    frequencies = np.asarray(counts) / rolls
    plt.hist(numbers, weights=frequencies, bins=len(NUMBERS))
    plt.title("Histogram of Dice Rolls")
    plt.xlabel("Dice Value")
    plt.ylabel("Frequency")
    plt.show()
    return frequencies, plt.gca()