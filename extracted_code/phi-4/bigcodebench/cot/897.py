import numpy as np
import matplotlib.pyplot as plt
import random

NUMBERS = list(range(1, 7))

def task_func(rolls, seed=None):
    if seed is not None:
        random.seed(seed)

    outcomes = np.zeros(6, dtype=int)
    for _ in range(rolls):
        roll = random.choice(NUMBERS)
        outcomes[roll - 1] += 1

    fig, ax = plt.subplots()
    ax.bar(NUMBERS, outcomes, color='skyblue', edgecolor='black')
    ax.set_title("Histogram of Dice Rolls")
    ax.set_xlabel("Dice Value")
    ax.set_ylabel("Frequency")
    ax.set_xticks(NUMBERS)

    return outcomes, ax