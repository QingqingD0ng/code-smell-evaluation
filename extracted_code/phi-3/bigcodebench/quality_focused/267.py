import numpy as np

from scipy import fftpack

import matplotlib.pyplot as plt


def task_func(data, sample_rate=8000):
    data['a'] = 1
    signal = np.array([value * np.sin(2 * np.pi * i / sample_rate) for i, value in enumerate(data.values())])
    fft_values = fftpack.fft(signal)
    plt.figure()
    plt.title('FFT of the signal')
    plt.plot(fft_values)
    plt.show()
    return fft_values, plt.gca()


# Example usage:

data = {'key1': 1, 'key2': 2, 'key3': 3}
fft, ax = task_func(data)