import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


def sine_wave(x, a, b, c, d):
    return a * np.sin(b * x + c) + d


def task_func(array_length=100, noise_level=0.2):
    x_data = np.linspace(0, 2 * np.pi, array_length)
    y_data = sine_wave(x_data, 1, 2, 0, 0)
    noise = np.random.normal(0, noise_level, array_length)
    y_data_noisy = y_data + noise

    popt, pcov = curve_fit(sine_wave, x_data, y_data_noisy)

    plt.figure()
    plt.plot(x_data, y_data_noisy, 'bo', label='Noisy data')
    plt.plot(x_data, sine_wave(x_data, *popt), 'r-', label='Fitted curve')
    plt.legend()
    plt.show()

    return plt