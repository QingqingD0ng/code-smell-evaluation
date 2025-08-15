import matplotlib.pyplot as plt
import scipy.optimize as optimize
import numpy as np

def task_func(array, target_value):
    # Extract rows where the first column matches the target value
    filtered_data = array[array[:, 0] == target_value]
    
    # Extract x (indices) and y (second column values) for fitting
    x_data = np.arange(len(filtered_data))
    y_data = filtered_data[:, 1]
    
    # Define the exponential decay function
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    # Fit the exponential decay function to the data
    popt, _ = optimize.curve_fit(exp_decay, x_data, y_data)
    
    # Plot the original data and the fitted curve
    fig, ax = plt.subplots()
    ax.scatter(x_data, y_data, label='Data')
    x_fit = np.linspace(0, len(filtered_data) - 1, 100)
    y_fit = exp_decay(x_fit, *popt)
    ax.plot(x_fit, y_fit, label='Fitted curve', color='red')
    ax.legend()
    
    return popt, ax