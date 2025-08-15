import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

def task_func(points, seed=0):
    np.random.seed(seed)
    
    jitter = np.random.normal(scale=1e-10, size=points.shape)
    points_jittered = points + jitter
    
    vor = Voronoi(points_jittered)
    
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange')
    
    ax.plot(points[:, 0], points[:, 1], 'b.')
    
    return vor, ax