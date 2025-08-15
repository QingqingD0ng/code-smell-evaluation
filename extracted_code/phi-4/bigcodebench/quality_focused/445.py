import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

def task_func(points, seed=0):
    np.random.seed(seed)
    points = np.array(points, dtype=float)
    jitter = np.random.normal(scale=1e-10, size=points.shape)
    points += jitter
    vor = Voronoi(points)
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange')
    ax.set_xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
    ax.set_ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
    return vor, ax

# Example usage:
# points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# vor, ax = task_func(points)