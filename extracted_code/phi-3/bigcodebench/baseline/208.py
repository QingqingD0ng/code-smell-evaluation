import numpy as np
import matplotlib.pyplot as plt

def task_func(elements, seed=0):
    if not isinstance(elements, int) or elements <= 0:
        raise ValueError("elements must be a positive integer")
        
    np.random.seed(seed)
    steps = np.random.choice([-1, 1], size=elements)
    walk = np.cumsum(steps)
    stats = {
        'count': elements,
       'mean': np.mean(walk),
       'std': np.std(walk, ddof=1),
       'min': np.min(walk),
        '5%': np.percentile(walk, 5),
        '25%': np.percentile(walk, 25),
        '50%': np.percentile(walk, 50),
        '75%': np.percentile(walk, 75),
        '95%': np.percentile(walk, 95),
       'max': np.max(walk)
    }
    
    plt.figure()
    plt.plot(walk)
    plt.title('Random Walk')
    plt.xlabel('Step')
    plt.ylabel('Position')
    
    return stats, plt.gca()