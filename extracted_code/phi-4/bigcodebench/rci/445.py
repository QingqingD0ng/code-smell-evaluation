import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

JITTER_STD_DEV = 0.01

def task_func(points, seed=0):
    if not isinstance(points, np.ndarray) or points.shape[1]!= 2:
        raise ValueError("points must be a numpy ndarray of shape (n_points, 2)")
    
    np.random.seed(seed)
    jittered_points = points + np.random.normal(0, JITTER_STD_DEV, points.shape)
    vor = Voronoi(jittered_points)
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax)
    return vor, ax, fig