import numpy as np
import matplotlib.pyplot as plt
import random

NUMBERS = list(range(1, 7))

def task_func(rolls, seed=None):
    if seed is not None:
        random.seed(seed)
    outcomes = np.zeros(6, dtype=int)
    for _ in range(rolls):
        outcome = random.choice(NUMBERS)
        outcomes[outcome - 1] += 1
    frequencies = outcomes / rolls
    ax = plt.hist(NUMBERS, weights=frequencies, bins=np.arange(0.5, 7.5, 1), alpha=0.7)
    ax.set_title('Histogram of Dice Rolls')
    ax.set_xlabel('Dice Value')
    ax.set_ylabel('Frequency')
    return outcomes, ax