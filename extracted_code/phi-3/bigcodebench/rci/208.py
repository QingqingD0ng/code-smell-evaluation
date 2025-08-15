import numpy as np
import matplotlib.pyplot as plt

def task_func(elements: int, seed: int = 0) -> dict:
    if not isinstance(elements, int) or elements <= 0:
        raise ValueError("elements must be a positive integer")

    np.random.seed(seed)
    steps = np.random.choice([-1, 1], size=elements)
    walk = np.cumsum(steps)
    stats = np.array([np.min(walk), np.percentile(walk, 5), np.percentile(walk, 25), 
                      np.percentile(walk, 50), np.percentile(walk, 75), np.percentile(walk, 95), 
                      np.max(walk), elements, np.mean(walk), np.std(walk, ddof=1)])
    stats_dict = dict(zip(('min', '5%', '25%','median', '75%', '95%','max', 'count','mean','std'), stats))

    fig, ax = plt.subplots()
    ax.plot(walk)
    ax.set_title('Random Walk')
    ax.set_xlabel('Step')
    ax.set_ylabel('Position')

    return stats_dict, ax