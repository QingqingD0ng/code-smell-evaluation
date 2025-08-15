import numpy as np

import matplotlib.pyplot as plt

import random


def task_func(rolls, seed=None):

    if seed is not None:

        random.seed(seed)

    results = [random.choice(NUMBERS) for _ in range(rolls)]

    frequency = np.bincount(results, minlength=len(NUMBERS))


    ax = plt.subplot(111)

    ax.bar(NUMBERS, frequency)

    ax.set_title('Histogram of Dice Rolls')

    ax.set_xlabel('Dice Value')

    ax.set_ylabel('Frequency')

    plt.show()


    return frequency, ax


# Example usage:

# outcomes, ax = task_func(10000, seed=0)

# print(outcomes)