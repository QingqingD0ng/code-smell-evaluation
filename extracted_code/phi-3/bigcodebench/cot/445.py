import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

def task_func(points, seed=0):
    np.random.seed(seed)
    jittered_points = points + np.random.uniform(-0.05, 0.05, points.shape)
    vor = Voronoi(jittered_points)
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax)
    ax.set_aspect('equal')
    plt.show()
    return vor, ax