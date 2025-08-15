import numpy as np
import matplotlib.pyplot as plt
import random

NUMBERS = list(range(1, 7))

def task_func(rolls, seed=None):
    if seed is not None:
        random.seed(seed)

    outcomes = np.zeros(len(NUMBERS))
    for _ in range(rolls):
        roll = random.choice(NUMBERS)
        outcomes[roll - 1] += 1

    histogram_axes = plt.subplot(1, 1, 1)
    plt.bar(NUMBERS, outcomes, tick_label=NUMBERS)
    plt.title("Histogram of Dice Rolls")
    plt.xlabel("Dice Value")
    plt.ylabel("Frequency")

    return outcomes, histogram_axes