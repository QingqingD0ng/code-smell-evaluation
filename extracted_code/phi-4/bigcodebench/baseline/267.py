import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def task_func(data, sample_rate=8000):
    data['a'] = 1
    keys = list(data.keys())
    values = list(data.values())
    signal = np.array(values)
    
    fft_result = fftpack.fft(signal)
    freq = fftpack.fftfreq(len(signal), 1/sample_rate)
    
    plt.figure()
    ax = plt.plot(freq, np.abs(fft_result))
    plt.title('FFT of the signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    
    return fft_result, ax