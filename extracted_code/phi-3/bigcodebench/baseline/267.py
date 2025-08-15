import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def task_func(data, sample_rate=8000):
    data['a'] = 1
    time = np.linspace(0, 1, sample_rate, endpoint=False)
    signal = np.sum([value * np.sin(2 * np.pi * frequency * time) for frequency, value in data.items()], axis=0)
    fft_result = fftpack.fft(signal)
    frequencies = fftpack.fftfreq(sample_rate, d=1/sample_rate)
    plt.figure()
    plt.plot(frequencies, np.abs(fft_result))
    plt.title('FFT of the signal')
    return fft_result, plt.gca()