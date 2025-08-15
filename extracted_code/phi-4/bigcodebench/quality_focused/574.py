from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

def sine_wave(x, amplitude, frequency, phase, offset):
    return amplitude * np.sin(frequency * x + phase) + offset

def task_func(array_length=100, noise_level=0.2):
    x = np.linspace(0, 4 * np.pi, array_length)
    y = 2.5 * np.sin(1.5 * x + 0.5) + 1
    noise = noise_level * np.random.normal(size=array_length)
    noisy_y = y + noise
    
    popt, _ = curve_fit(sine_wave, x, noisy_y, p0=[2, 1, 0, 1])
    
    fitted_y = sine_wave(x, *popt)
    
    plt.plot(x, noisy_y, 'b-', label='Noisy Data')
    plt.plot(x, fitted_y, 'r--', label='Fitted Curve')
    plt.legend()
    plt.show()