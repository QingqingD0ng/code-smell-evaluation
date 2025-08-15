import numpy as np
import matplotlib.pyplot as plt
import random

NUMBERS = list(range(1, 7))

def task_func(rolls, seed=None):
    if seed is not None:
        random.seed(seed)
    outcomes = [random.choice(NUMBERS) for _ in range(rolls)]
    frequency = np.array([outcomes.count(i) for i in NUMBERS])
    
    fig, ax = plt.subplots()
    ax.hist(outcomes, bins=range(1, 8), rwidth=0.8, align='left', color='skyblue', edgecolor='black')
    ax.set_title("Histogram of Dice Rolls")
    ax.set_xlabel("Dice Value")
    ax.set_ylabel("Frequency")
    
    return frequency, ax