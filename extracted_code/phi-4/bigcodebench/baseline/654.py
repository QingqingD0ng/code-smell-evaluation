import matplotlib.pyplot as plt
import scipy.optimize as optimize
import numpy as np

def task_func(array, target_value):
    filtered_data = array[array[:, 0] == target_value]
    if filtered_data.size == 0:
        return None, None
    x_data = np.arange(filtered_data.shape[0])
    y_data = filtered_data[:, 1]

    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    popt, _ = optimize.curve_fit(exp_decay, x_data, y_data, p0=(1, 1, 1))

    plt.figure()
    ax = plt.gca()
    ax.plot(x_data, y_data, 'bo', label='Data')
    ax.plot(x_data, exp_decay(x_data, *popt), 'r-', label='Fit')
    ax.legend()

    return popt, ax