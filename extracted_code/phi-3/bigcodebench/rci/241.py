import numpy as np

import matplotlib.pyplot as plt

from sklearn import preprocessing


def normalize_and_plot(data):

    values = np.array([value for _, value in data]).reshape(-1, 1)

    min_max_scaler = preprocessing.MinMaxScaler()

    normalized_values = min_max_scaler.fit_transform(values).flatten()


    plt.figure()

    plt.plot(values, label='Original')

    plt.plot(normalized_values, label='Normalized')

    plt.title('Original vs. Normalized Data')

    plt.legend()

    plt.show()


    return values.flatten(), normalized_values


# Example usage:

original_data = [('a', 1), ('b', 2), ('c', 3), ('d', 4)]

original_values, normalized_values = normalize_and_plot(original_data)