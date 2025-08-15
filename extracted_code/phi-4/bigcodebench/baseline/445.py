import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

def task_func(points, seed=0):
    np.random.seed(seed)
    points = np.array(points) + np.random.normal(0, 0.01, points.shape)
    vor = Voronoi(points)
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax)
    return vor, ax