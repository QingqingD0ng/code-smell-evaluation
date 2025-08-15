import numpy as np
import matplotlib.pyplot as plt
import random

NUMBERS = list(range(1, 7))

def task_func(rolls, seed=None):
    if seed is not None:
        random.seed(seed)
    
    results = [random.choice(NUMBERS) for _ in range(rolls)]
    frequency = np.array([results.count(i) for i in NUMBERS])
    
    fig, ax = plt.subplots()
    ax.bar(NUMBERS, frequency)
    ax.set_title("Histogram of Dice Rolls")
    ax.set_xlabel("Dice Value")
    ax.set_ylabel("Frequency")
    
    return frequency, ax

# Example usage
if __name__ == "__main__":
    random.seed(0)
    outcomes, ax = task_func(10000)
    print(outcomes)
    plt.show()

    random.seed(10)
    outcomes, ax = task_func(100)
    print(outcomes)
    plt.show()