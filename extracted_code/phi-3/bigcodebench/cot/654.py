import matplotlib.pyplot as plt

import scipy.optimize as optimize

import numpy as np


def task_func(array, target_value):
    x_data = array[:, 0]
    y_data = array[:, 1]
    mask = x_data == target_value
    x_filtered = x_data[mask]
    y_filtered = y_data[mask]
    popt, pcov = optimize.curve_fit(lambda x, A, tau: A * np.exp(-x / tau), x_filtered, y_filtered)
    plt.scatter(x_filtered, y_filtered, label='Data')
    plt.plot(x_filtered, popt[0] * np.exp(-x_filtered / popt[1]), label='Fit', color='red')
    plt.legend()
    fig, ax = plt.subplots()
    return popt, ax


# Example usage:

# import numpy as np

# array = np.array([[1, 2], [1, 3], [1, 4], [2, 5], [2, 6]])
# target = 1
# params, ax = task_func(array, target)
# print(len(params))