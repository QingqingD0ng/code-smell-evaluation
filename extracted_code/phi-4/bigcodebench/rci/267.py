import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def task_func(data, sample_rate=8000):
    if not isinstance(data, dict) or not all(isinstance(v, (int, float)) for v in data.values()):
        raise ValueError("Input data must be a dictionary with numeric values.")
    
    if not data:
        raise ValueError("Input data dictionary is empty.")
    
    data['a'] = 1
    values = np.array(list(data.values()))
    
    if len(values) == 0:
        raise ValueError("Signal derived from data is empty.")
    
    fft_result = fftpack.fft(values)
    freq = fftpack.fftfreq(len(values), 1/sample_rate)
    
    plt.figure()
    ax = plt.gca()
    ax.plot(freq, np.abs(fft_result))
    ax.set_title('FFT of the signal')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    
    plt.show()
    
    return fft_result, ax