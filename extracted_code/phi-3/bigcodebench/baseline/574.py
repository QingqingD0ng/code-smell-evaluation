import numpy as np

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt


def sine_wave_with_noise(x, amplitude, frequency, phase, offset):
    return amplitude * np.sin(frequency * x + phase) + offset


def task_func(array_length=100, noise_level=0.2):
    x_data = np.linspace(0, 2 * np.pi, array_length)
    y_data = sine_wave_with_noise(x_data, amplitude=1, frequency=1, phase=0, offset=0)
    noise = noise_level * np.random.normal(size=x_data.size)
    y_noisy = y_data + noise

    popt, pcov = curve_fit(sine_wave_with_noise, x_data, y_noisy, p0=[1, 1, 0, 0])

    plt.figure()
    plt.plot(x_data, y_noisy, 'b-', label='Noisy Data')
    plt.plot(x_data, sine_wave_with_noise(x_data, *popt), 'r--', label='Fitted Curve')
    plt.legend()
    return plt.gca()


# Example usage:

ax = task_func(100, 0.2)