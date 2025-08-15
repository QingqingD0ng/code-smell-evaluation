import scipy.optimize as opt

import matplotlib.pyplot as plt

import numpy as np


def task_func(array_length=100, noise_level=0.2):
    x = np.linspace(0, 2 * np.pi, array_length)
    y = np.sin(x)
    noise = np.random.normal(0, noise_level, array_length)
    y_noisy = y + noise
    
    def sine_curve(x, A, B, C, D):
        return A * np.sin(B * x + C) + D
    
    params, _ = opt.curve_fit(sine_curve, x, y_noisy)
    
    plt.plot(x, y_noisy, 'o', label='Noisy Data')
    plt.plot(x, sine_curve(x, *params), label='Fitted Curve')
    
    plt.legend()
    plt.show()


# Example usage

ax = task_func(100, 0.2)