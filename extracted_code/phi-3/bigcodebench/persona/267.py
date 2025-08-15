import numpy as np

from scipy import fftpack

import matplotlib.pyplot as plt


def task_func(data, sample_rate=8000):
    data['a'] = 1
    # Assuming the values in data represent samples of a sine wave with the frequency corresponding to the index
    time = np.arange(len(data)) / sample_rate
    signal = np.sum(np.array(list(data.values())) * np.sin(2 * np.pi * time))
    fft_result = fftpack.fft(signal)
    frequencies = fftpack.fftfreq(len(signal), d=1/sample_rate)

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, np.abs(fft_result))
    plt.title('FFT of the signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    ax = plt.gca()
    plt.show()

    return np.abs(fft_result), ax


# Example usage:

data = {'key1': 1, 'key2': 2, 'key3': 3}
fft, ax = task_func(data)