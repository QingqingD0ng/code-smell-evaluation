import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

def validate_input(points):
    if not isinstance(points, np.ndarray) or points.ndim!= 2 or points.shape[1]!= 2:
        raise ValueError('Points must be a numpy array of shape (n_points, 2)')

def jitter_points(points, jitter_range=0.05):
    jitter = np.random.uniform(-jitter_range, jitter_range, points.shape)
    return points + jitter

def create_voronoi(points, seed):
    np.random.seed(seed)
    jittered_points = jitter_points(points)
    vor = Voronoi(jittered_points)
    return vor

def plot_voronoi(vor):
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax)
    ax.autoscale_view()
    plt.show()

def task_func(points, seed=0):
    validate_input(points)
    vor = create_voronoi(points, seed)
    plot_voronoi(vor)