import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def task_func(data, sample_rate=8000):
    # Update data dictionary with a new key 'a'
    data['a'] = 1

    # Create a signal from the dictionary values
    signal = np.array(list(data.values()))

    # Perform FFT on the signal
    fft_result = fftpack.fft(signal)

    # Generate frequency values for the x-axis
    freqs = fftpack.fftfreq(len(signal), 1/sample_rate)

    # Plot the FFT results
    plt.figure()
    ax = plt.plot(freqs, np.abs(fft_result))
    plt.title('FFT of the signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid()

    # Return the FFT values and the plot axes
    return fft_result, ax