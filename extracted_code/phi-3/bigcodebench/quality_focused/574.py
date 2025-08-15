import numpy as np

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt


def task_func(array_length=100, noise_level=0.2):
    x_values = np.linspace(0, 2 * np.pi, array_length)
    y_values = np.sin(x_values)
    noisy_y = y_values + noise_level * np.random.randn(array_length)

    def sine_function(x, A, omega, phi):
        return A * np.sin(omega * x + phi)

    params, _ = curve_fit(sine_function, x_values, noisy_y)

    plt.scatter(x_values, noisy_y, label='Noisy sine wave')
    plt.plot(x_values, sine_function(x_values, *params), label='Fitted curve', color='red')
    plt.legend()
    plt.show()


# Example usage:

ax = task_func(100, 0.2)