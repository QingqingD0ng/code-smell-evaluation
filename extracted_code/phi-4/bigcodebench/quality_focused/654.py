import matplotlib.pyplot as plt
import scipy.optimize as optimize
import numpy as np

def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def task_func(array, target_value):
    filtered_array = array[array[:, 0] == target_value]
    x_data = np.arange(len(filtered_array))
    y_data = filtered_array[:, 1]

    popt, _ = optimize.curve_fit(exp_decay, x_data, y_data, p0=(1, 1, 1))

    ax = plt.figure().add_subplot()
    ax.plot(x_data, y_data, 'o', label='Data')
    ax.plot(x_data, exp_decay(x_data, *popt), '-', label='Fitted function')
    ax.legend()
    plt.show()

    return popt, ax