import numpy as np

from scipy import fftpack

import matplotlib.pyplot as plt


def task_func(data, sample_rate=8000):
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary containing frequencies and corresponding values.")

    data['a'] = 1
    time = np.linspace(0, 1, sample_rate, endpoint=False)
    frequencies = list(data.keys())
    values = list(data.values())
    signal = np.sum([value * np.sin(2 * np.pi * frequency * time) for frequency, value in zip(frequencies, values)], axis=0)
    fft_result = fftpack.fft(signal)
    ax = plt.gca()
    ax.plot(fftpack.fftfreq(sample_rate, d=1/sample_rate), np.abs(fft_result))
    ax.set_title('FFT of the signal')
    plt.show()
    return fft_result, ax