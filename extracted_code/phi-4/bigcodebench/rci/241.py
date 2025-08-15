import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def extract_numeric_values(original):
    return np.array([item[1] for item in original])

def normalize_array(arr):
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(arr.reshape(-1, 1)).flatten()

def plot_original_vs_normalized(original, normalized):
    fig, ax = plt.subplots()
    ax.plot(original, label='Original')
    ax.plot(normalized, label='Normalized')
    ax.set_title('Original vs. Normalized Data')
    ax.legend()
    return fig, ax

def task_func(original):
    original_array = extract_numeric_values(original)
    normalized_array = normalize_array(original_array)
    fig, ax = plot_original_vs_normalized(original_array, normalized_array)
    return original_array, normalized_array, fig, ax