import numpy as np
import os
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

def task_func(L, M, N, audio_file):
    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"The specified audio file does not exist: {audio_file}")
    
    data, sr = librosa.load(audio_file)
    spl = 20 * np.log10(np.sqrt(np.mean(data**2)))
    normalized_matrix = np.clip(np.array(L).reshape(M, N) / np.max(np.array(L).reshape(M, N)), 0, 1) * 10**(spl / 20)

    fig, ax = plt.subplots()
    img = ax.imshow(normalized_matrix, aspect='auto', origin='lower', cmap='inferno')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    ax.set_title('Spectrogram with SPL Adjustment')
    plt.colorbar(img, ax=ax)
    plt.show()
    
    return normalized_matrix