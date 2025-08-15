import numpy as np
import os
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

def task_func(L, M, N, audio_file):
    if not os.path.exists(audio_file):
        raise FileNotFoundError("The specified audio file does not exist.")
    
    data, sr = librosa.load(audio_file, sr=None)
    spl = 20 * np.log10(np.sqrt(np.mean(data**2)))
    normalized_data = data / (10**(spl/20))
    
    matrix = np.array(L).reshape(M, N)
    matrix = matrix - np.mean(matrix)
    matrix = matrix / np.std(matrix)
    
    matrix = matrix * (np.max(np.abs(normalized_data)) / np.max(np.abs(matrix)))
    
    fig, ax = plt.subplots()
    Pxx, freqs, bins, im = ax.specgram(matrix, Fs=sr)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_title(f'Spectrogram with SPL: {spl:.2f} dB')
    return matrix, fig