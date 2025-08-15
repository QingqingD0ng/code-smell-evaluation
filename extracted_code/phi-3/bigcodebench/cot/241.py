import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def task_func(original):
    arr = np.array(original)[:, 1].reshape(-1, 1)
    scaler = preprocessing.StandardScaler()
    norm_arr = scaler.fit_transform(arr)
    
    plt.figure()
    plt.title('Original vs. Normalized Data')
    plt.plot(arr, label='Original', marker='o')
    plt.plot(norm_arr, label='Normalized', marker='x')
    plt.legend()
    plt.grid(True)

    return arr, norm_arr, plt

original = [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
arr, norm_arr, ax = task_func(original)
ax.show()