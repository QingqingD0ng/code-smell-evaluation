import numpy as np
import os
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

def task_func(L, M, N, audio_file):
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"The file {audio_file} does not exist.")

    # Create MxN matrix from list L
    matrix = np.array(L[:M*N]).reshape(M, N)

    # Calculate SPL from audio file
    data, _ = sf.read(audio_file)
    spl = 20 * np.log10(np.sqrt(np.mean(data**2)))

    # Normalize matrix by SPL
    normalized_matrix = matrix / (10**(spl / 20))

    # Generate spectrogram
    S = librosa.stft(normalized_matrix.flatten())
    D = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    fig, ax = plt.subplots()
    img = librosa.display.specshow(D, sr=22050, x_axis='time', y_axis='log', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()

    return normalized_matrix, fig