import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def task_func(data, sample_rate=8000):
    data['a'] = 1
    signal = np.array(list(data.values()))
    fft_result = fftpack.fft(signal)
    frequencies = fftpack.fftfreq(len(signal), 1/sample_rate)
    
    plt.figure()
    ax = plt.gca()
    ax.plot(frequencies, np.abs(fft_result))
    ax.set_title('FFT of the signal')
    plt.show()
    
    return fft_result, ax