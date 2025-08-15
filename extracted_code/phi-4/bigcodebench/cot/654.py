import matplotlib.pyplot as plt
import scipy.optimize as optimize
import numpy as np

def task_func(array, target_value):
    filtered_array = array[array[:, 0] == target_value]
    x_data = filtered_array[:, 1]
    y_data = np.arange(len(x_data))

    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    popt, _ = optimize.curve_fit(exp_decay, x_data, y_data)

    plt.figure()
    ax = plt.gca()
    ax.scatter(x_data, y_data, label='Data')
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = exp_decay(x_fit, *popt)
    ax.plot(x_fit, y_fit, label='Fitted curve', color='red')
    ax.legend()

    return popt, ax