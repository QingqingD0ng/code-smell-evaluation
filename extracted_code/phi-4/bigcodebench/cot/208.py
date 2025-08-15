import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def task_func(elements, seed=0):
    if not isinstance(elements, int) or elements <= 0:
        raise ValueError("elements must be a positive integer")

    np.random.seed(seed)
    steps = np.random.choice([-1, 1], size=elements)
    random_walk = np.cumsum(steps)

    plt.figure()
    plt.plot(random_walk, label='Random Walk')
    plt.xlabel('Step')
    plt.ylabel('Position')
    plt.title('Random Walk')
    plt.legend()
    ax = plt.gca()
    plt.show()

    stats = pd.Series(random_walk).describe(percentiles=[.05,.25,.5,.75,.95])
    stats_dict = {key: value for key, value in stats.items()}

    return stats_dict, ax