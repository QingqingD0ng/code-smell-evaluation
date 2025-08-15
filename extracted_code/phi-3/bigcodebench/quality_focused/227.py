import numpy as np
import os
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from math import log10, sqrt

def normalize_spl(audio_data):
    mean_square = np.mean(audio_data ** 2)
    spl = 20 * log10(sqrt(mean_square))
    return audio_data / sqrt(mean_square) * 10 ** (spl / 20)

def create_matrix(L, M, N):
    return np.array(L).reshape(M, N)

def generate_spectrogram(matrix, sr):
    S = librosa.feature.melspectrogram(S=matrix, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    return plt

def task_func(L, M, N, audio_file):
    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"Audio file {audio_file} not found.")
    
    data, sr = sf.read(audio_file)
    normalized_data = normalize_spl(data)
    matrix = create_matrix(normalized_data, M, N)
    spectrogram = generate_spectrogram(matrix, sr)
    return matrix, spectrogram

# Example usage:
# matrix, spectrogram = task_func([i for i in range(100)], 10, 10, 'audio.wav')
# spectrogram.show()