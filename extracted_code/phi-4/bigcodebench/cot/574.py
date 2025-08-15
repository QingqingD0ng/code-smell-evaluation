from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

def sine_wave(x, a, b, c):
    return a * np.sin(b * x + c)

def task_func(array_length=100, noise_level=0.2):
    x_data = np.linspace(0, 2 * np.pi, array_length)
    y_data = np.sin(x_data) + noise_level * np.random.randn(array_length)

    popt, _ = curve_fit(sine_wave, x_data, y_data, p0=[1, 1, 0])

    y_fit = sine_wave(x_data, *popt)

    fig, ax = plt.subplots()
    ax.plot(x_data, y_data, 'b.', label='Noisy Data')
    ax.plot(x_data, y_fit, 'r-', label='Fitted Curve')
    ax.legend()
    plt.show()

    return ax