import numpy as np
import matplotlib.pyplot as plt
import math

def task_func(frequency, sample_size=10000):
    if frequency < 0:
        raise ValueError("Frequency must be non-negative.")
    if not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("Sample size must be a positive integer.")

    TWO_PI = 2 * math.pi
    t = np.linspace(0, TWO_PI, sample_size)
    sine_wave = np.sin(frequency * t)
    cosine_wave = np.cos(frequency * t)

    fig, ax = plt.subplots()
    ax.plot(t, sine_wave, label='Sine Wave')
    ax.plot(t, cosine_wave, label='Cosine Wave')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Sine and Cosine Waves with Frequency {frequency}')
    ax.legend(loc='upper right')

    return fig, ax