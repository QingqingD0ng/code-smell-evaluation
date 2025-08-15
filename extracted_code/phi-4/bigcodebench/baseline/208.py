import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def task_func(elements, seed=0):
    if elements <= 0:
        raise ValueError("elements must be a positive integer.")
    
    np.random.seed(seed)
    steps = np.random.choice([-1, 1], size=elements)
    random_walk = np.cumsum(steps)
    
    stats = pd.Series(random_walk).describe(percentiles=[.05,.25,.5,.75,.95]).to_dict()
    stats['count'] = stats.pop('count')
    
    plt.figure()
    ax = plt.plot(random_walk)
    plt.title('Random Walk')
    plt.xlabel('Step')
    plt.ylabel('Position')
    
    return stats, ax