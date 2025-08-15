import numpy as np
import os
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

def task_func(L, M, N, audio_file):
    if not os.path.isfile(audio_file):
        raise FileNotFoundError("Audio file not found.")
    
    data, sr = librosa.load(audio_file)
    SPL = 20 * np.log10(np.sqrt(np.mean(data**2)))
    normalized_matrix = np.array(L).reshape(M, N) / SPL
    
    fig = plt.figure()
    SPL_adjusted = normalized_matrix * SPL
    D = librosa.amplitude_to_db(SPL_adjusted, ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()
    
    return normalized_matrix

# Example usage:
# matrix = task_func([i for i in range(100)], 10, 10, 'audio.wav')
# print(matrix.shape)