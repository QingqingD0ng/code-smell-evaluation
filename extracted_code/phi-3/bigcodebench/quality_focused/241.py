import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def task_func(original):
    arr = np.array(original).T[1]
    scaler = preprocessing.MinMaxScaler()
    norm_arr = scaler.fit_transform(arr.reshape(-1, 1)).flatten()
    
    fig, ax = plt.subplots()
    ax.plot(arr, label='Original')
    ax.plot(norm_arr, label='Normalized')
    ax.set_title('Original vs. Normalized Data')
    ax.legend()
    
    plt.show()
    
    return arr, norm_arr, ax