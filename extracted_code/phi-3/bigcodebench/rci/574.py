import numpy as np

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt


TWO_PI = 2 * np.pi


def generate_noisy_sine_wave(array_length: int = 100, noise_level: float = 0.2,
                             initial_params: list = [1.0, 1.0, 0.0, 0.0]) -> plt.Axes:

    x_data = np.linspace(0, TWO_PI, array_length)

    y_data = np.sin(x_data * initial_params[1] + initial_params[2]) + initial_params[3]

    noise = noise_level * np.random.normal(size=x_data.size)

    y_noisy = y_data + noise


    popt, pcov = curve_fit(lambda x, A, f, p, c: A * np.sin(f * x + p) + c, x_data, y_noisy, p0=initial_params)


    fig, ax = plt.subplots()

    ax.plot(x_data, y_noisy, 'b-', label='Noisy Data')

    ax.plot(x_data, popt[0] * np.sin(popt[1] * x_data + popt[2]) + popt[3], 'r--', label='Fitted Curve')

    ax.set_title('Noisy Sine Wave and Fitted Curve')

    ax.set_xlabel('x')

    ax.set_ylabel('y')

    ax.grid(True)

    ax.legend()

    return ax


# Example usage:

plt.show()