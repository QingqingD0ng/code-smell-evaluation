import numpy as np
import matplotlib.pyplot as plt
import math

def task_func(frequency, sample_size=10000):
    if frequency < 0:
        raise ValueError("Frequency must be non-negative")
    if sample_size <= 0:
        raise ValueError("Sample size must be positive")
    
    t = np.linspace(0, 2 * math.pi, sample_size)
    sine_wave = np.sin(frequency * t)
    cosine_wave = np.cos(frequency * t)
    
    fig, ax = plt.subplots()
    ax.plot(t, sine_wave, label='Sine wave')
    ax.plot(t, cosine_wave, label='Cosine wave')
    ax.legend()
    
    return fig, ax