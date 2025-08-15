import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Dict

def generate_random_walk(elements: int, seed: int = 0) -> np.ndarray:
    np.random.seed(seed)
    steps = np.random.choice([-1, 1], size=elements)
    return np.cumsum(steps)

def calculate_statistics(random_walk: np.ndarray) -> Dict[str, float]:
    stats = pd.Series(random_walk).describe(percentiles=[.05,.25,.5,.75,.95]).to_dict()
    stats['count'] = stats.pop('count')
    return stats

def plot_random_walk(random_walk: np.ndarray) -> plt.Axes:
    plt.figure()
    ax = plt.plot(random_walk)
    plt.title('Random Walk')
    plt.xlabel('Step')
    plt.ylabel('Position')
    return ax

def task_func(elements: int, seed: int = 0) -> Tuple[Dict[str, float], plt.Axes]:
    if elements <= 0:
        raise ValueError("elements must be a positive integer.")
    
    random_walk = generate_random_walk(elements, seed)
    stats = calculate_statistics(random_walk)
    ax = plot_random_walk(random_walk)
    
    return stats, ax