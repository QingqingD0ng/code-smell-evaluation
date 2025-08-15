import numpy as np

import matplotlib.pyplot as plt

import math


FULL_PERIOD = 2 * math.pi


def task_func(frequency, sample_size=10000):
    if frequency < 0:
        raise ValueError("Frequency must be non-negative")
    if sample_size <= 0:
        raise ValueError("Sample size must be positive")
    
    t = np.linspace(0, FULL_PERIOD, sample_size)
    sine_wave = np.sin(frequency * t)
    cosine_wave = np.cos(frequency * t)
    
    fig, ax = plt.subplots()
    ax.plot(t, sine_wave, label='Sine wave')
    ax.plot(t, cosine_wave, label='Cosine wave')
    ax.legend()
    ax.grid(True)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Sine and Cosine Waves at {frequency} Hz')
    plt.tight_layout()
    
    return fig, ax