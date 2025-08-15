import matplotlib.pyplot as plt

import scipy.optimize as optimize

import numpy as np


def task_func(array, target_value):
    # Filter the array for rows where the first value matches target_value
    filtered_data = array[array[:, 0] == target_value]
    
    # Extract x (indices) and y (values) for fitting
    x = np.arange(len(filtered_data))
    y = filtered_data[:, 1]
    
    # Define the exponential decay function to fit
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    # Perform the curve fitting
    popt, _ = optimize.curve_fit(exp_decay, x, y)
    
    # Create a plot
    ax = plt.subplot(111)
    ax.plot(x, y, 'ko', label='Original data')
    ax.plot(x, exp_decay(x, *popt), 'r-', label='Fitted curve')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.legend()
    
    return popt, ax


# Example usage

import numpy as np

array = np.array([[1, 2], [1, 3], [1, 4], [2, 5], [2, 6]])
target = 1
params, ax = task_func(array, target)
print(params)

plt.show()