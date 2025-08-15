from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

def sine_wave(x, amplitude, frequency, phase, offset):
    return amplitude * np.sin(frequency * x + phase) + offset

def task_func(array_length=100, noise_level=0.2):
    x_data = np.linspace(0, 2 * np.pi, array_length)
    y_data = 2 * np.sin(1.5 * x_data + 0.5) + np.random.normal(0, noise_level, array_length)

    initial_guess = [2, 1.5, 0.5, 0]
    params, _ = curve_fit(sine_wave, x_data, y_data, p0=initial_guess)

    y_fit = sine_wave(x_data, *params)

    plt.figure()
    plt.plot(x_data, y_data, 'b-', label='Noisy data')
    plt.plot(x_data, y_fit, 'r-', label='Fitted curve')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Noisy Sine Wave and Fitted Curve')
    plt.show()

    return plt.gca()

# Example usage
# ax = task_func(100, 0.2)