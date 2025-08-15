import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def task_func(elements, seed=0):
    if not isinstance(elements, int) or elements <= 0:
        raise ValueError("elements must be a positive integer")
    
    np.random.seed(seed)
    steps = np.random.choice([-1, 1], size=elements)
    path = np.cumsum(steps)
    
    stats = pd.DataFrame(path, columns=['Step']).describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).transpose().to_dict()
    stats['count'] = elements
    
    fig, ax = plt.subplots()
    ax.plot(path, label='Random Walk')
    ax.set_title('Random Walk')
    ax.set_xlabel('Step')
    ax.set_ylabel('Position')
    ax.legend()
    
    return stats, ax

# Example call
stats, ax = task_func(1000)
plt.show()  # This line is for demonstration purposes to show the plot. It should be outside the function if the plot is not needed programmatically.