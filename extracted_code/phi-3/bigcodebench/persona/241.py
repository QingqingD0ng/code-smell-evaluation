import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing


def task_func(original):
    arr = np.array(original).T[1]
    norm_arr = preprocessing.normalize([arr])[0]
    plt.figure()
    plt.plot(original[0], arr, label='Original')
    plt.plot(original[0], norm_arr, label='Normalized')
    plt.title('Original vs. Normalized Data')
    plt.legend()
    return arr, norm_arr, plt.gca()


# Example usage:

original = [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
arr, norm_arr, ax = task_func(original)