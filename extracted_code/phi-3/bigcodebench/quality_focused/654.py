import matplotlib.pyplot as plt

import scipy.optimize as optimize

import numpy as np


def task_func(array, target_value):
    indices = np.where(array[:, 0] == target_value)
    data = array[indices][:, 1]
    p0 = [1, 0, 1]  # Initial guess for the parameters
    
    def model(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    popt, pcov = optimize.curve_fit(model, np.arange(len(data)), data, p0=p0)
    
    plt.scatter(np.arange(len(data)), data, label='Data')
    plt.plot(np.arange(len(data)), model(np.arange(len(data)), *popt), label='Fit', color='red')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Value')
    ax = plt.gca()
    
    return popt, ax


# Example usage:

array = np.array([[1, 2], [1, 3], [1, 4], [2, 5], [2, 6]])
target = 1
params, ax = task_func(array, target)
print(len(params))