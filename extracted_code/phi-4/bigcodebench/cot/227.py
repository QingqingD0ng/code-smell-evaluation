import numpy as np
import os
import soundfile as sf
import matplotlib.pyplot as plt

def task_func(L, M, N, audio_file):
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file '{audio_file}' not found.")
    
    matrix = np.array(L[:M*N]).reshape(M, N)
    
    data, _ = sf.read(audio_file)
    
    spl = 20 * np.log10(np.sqrt(np.mean(data**2)))
    
    normalized_matrix = matrix / (10**(spl / 20))
    
    plt.figure()
    plt.specgram(normalized_matrix.flatten(), NFFT=256, Fs=2, scale='dB')
    plt.title('Spectrogram')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    
    return normalized_matrix, plt.gcf()