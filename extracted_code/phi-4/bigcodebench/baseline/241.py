import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def task_func(original):
    arr = np.array([item[1] for item in original])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_arr = min_max_scaler.fit_transform(arr.reshape(-1, 1)).flatten()

    fig, ax = plt.subplots()
    ax.plot(arr, label='Original')
    ax.plot(norm_arr, label='Normalized')
    ax.set_title('Original vs. Normalized Data')
    ax.legend()

    return arr, norm_arr, ax