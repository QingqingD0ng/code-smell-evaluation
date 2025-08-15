import matplotlib.pyplot as plt

import scipy.optimize as optimize

import numpy as np


def fit_exponential_decay(array, target_value):
    data_points = array[array[:, 0] == target_value]
    if data_points.size == 0:
        raise ValueError("No data points available for the specified target value.")
    indices, values = data_points[:, 0], data_points[:, 1]
    
    def decay_function(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    params, _ = optimize.curve_fit(decay_function, indices, values)
    
    fig, ax = plt.subplots()
    ax.plot(indices, values, 'o', label='Original data')
    ax.plot(indices, decay_function(indices, *params), 'r-', label='Fitted curve')
    ax.set_xlabel('Indices')
    ax.set_ylabel('Values')
    ax.legend()
    
    plt.show()
    
    return params