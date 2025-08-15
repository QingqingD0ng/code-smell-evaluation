import numpy as np
import matplotlib.pyplot as plt
import math

def task_func(frequency, sample_size=10000):
    if frequency < 0:
        raise ValueError("Frequency must be non-negative.")
    if sample_size <= 0:
        raise ValueError("Sample size must be positive.")
    
    t = np.linspace(0, 2 * np.pi, sample_size)
    sine_wave = np.sin(frequency * t)
    cosine_wave = np.cos(frequency * t)
    
    plt.figure(figsize=(10, 4))
    plt.plot(t, sine_wave, label='Sine Wave')
    plt.plot(t, cosine_wave, label='Cosine Wave')
    plt.title(f'Sine and Cosine Waves with Frequency {frequency}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return plt.gcf(), plt.gca()