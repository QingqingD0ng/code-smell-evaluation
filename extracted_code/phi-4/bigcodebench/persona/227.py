import numpy as np
import os
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

def task_func(L, M, N, audio_file):
    # Check if the audio file exists
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"The file {audio_file} does not exist.")
    
    # Create MxN matrix from the list L
    matrix = np.array(L[:M*N]).reshape(M, N)
    
    # Read audio data
    data, sr = sf.read(audio_file)
    
    # Calculate Sound Pressure Level (SPL)
    spl = 20 * np.log10(np.sqrt(np.mean(data**2)))
    
    # Normalize the matrix based on SPL
    normalized_matrix = matrix / spl
    
    # Generate spectrogram
    S = np.abs(librosa.stft(normalized_matrix.flatten()))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    
    # Create figure for spectrogram
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    return normalized_matrix, fig