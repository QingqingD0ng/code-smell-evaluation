import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

def task_func(points, seed=0):
    np.random.seed(seed)
    
    if not isinstance(points, np.ndarray) or points.shape[1]!= 2:
        raise ValueError("Input must be a numpy ndarray of shape (n_points, 2)")
    
    # Add jittering to the points
    jitter = np.random.normal(scale=0.01, size=points.shape)
    points_jittered = points + jitter
    
    # Compute Voronoi diagram
    vor = Voronoi(points_jittered)
    
    # Plot Voronoi diagram
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)
    ax.scatter(points[:, 0], points[:, 1], color='red')  # Original points
    
    return vor, ax