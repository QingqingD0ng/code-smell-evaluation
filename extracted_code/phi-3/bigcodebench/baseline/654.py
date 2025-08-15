import matplotlib.pyplot as plt

import scipy.optimize as optimize

import numpy as np


def task_func(array, target_value):
    filtered_data = array[array[:, 0] == target_value]
    if filtered_data.size == 0:
        raise ValueError("No data points found for the target value.")
    x_data, y_data = filtered_data[:, 0], filtered_data[:, 1]
    
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    popt, pcov = optimize.curve_fit(exp_decay, x_data, y_data)
    
    ax = plt.subplot(111)
    ax.plot(x_data, y_data, 'o', label='Original data')
    ax.plot(x_data, exp_decay(x_data, *popt), 'r-', label='Fitted curve')
    ax.set_xlabel('Indices')
    ax.set_ylabel('Values')
    ax.legend()
    
    return popt, ax