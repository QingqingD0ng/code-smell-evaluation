import numpy as np

from scipy.spatial import Voronoi, voronoi_plot_2d

import matplotlib.pyplot as plt


def task_func(points, seed=0):

    np.random.seed(seed)

    if not isinstance(points, np.ndarray) or points.shape[1]!= 2:

        raise ValueError("Input points must be a numpy ndarray of shape (n_points, 2)")


    vor = Voronoi(points)

    fig, ax = plt.subplots()

    voronoi_plot_2d(vor, ax=ax)

    ax.plot(points[:, 0], points[:, 1], 'o')

    plt.show()


    return vor, ax