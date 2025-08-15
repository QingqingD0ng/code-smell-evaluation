import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def task_func(data, sample_rate=8000):
    data['a'] = 1
    signal = np.array(list(data.values()))
    fft_result = fftpack.fft(signal)
    freqs = fftpack.fftfreq(len(signal), 1/sample_rate)
    
    plt.figure()
    ax = plt.plot(freqs, np.abs(fft_result))
    plt.title('FFT of the signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()
    
    return fft_result, ax