import numpy as np
import os
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

def task_func(L, M, N, audio_file):
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"The file {audio_file} does not exist.")
    
    data, sr = sf.read(audio_file)
    spl = 20 * np.log10(np.sqrt(np.mean(data**2)))
    
    matrix = np.array(L[:M*N]).reshape(M, N)
    normalized_matrix = matrix / (10**(spl / 20))
    
    D = librosa.stft(normalized_matrix.flatten(), n_fft=2048, hop_length=512)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='log', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    return normalized_matrix, fig