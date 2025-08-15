import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def task_func(data, sample_rate=8000):
    data['a'] = 1
    signal = np.array([d for d in data.values()])
    fft_result = fftpack.fft(signal)
    ax = plt.subplot(111)
    ax.plot(np.abs(fft_result))
    plt.title('FFT of the signal')
    plt.show()
    return np.abs(fft_result), ax