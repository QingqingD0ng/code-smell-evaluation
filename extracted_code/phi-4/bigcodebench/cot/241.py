import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def task_func(original):
    data = np.array([item[1] for item in original])
    normalized_data = preprocessing.MinMaxScaler().fit_transform(data.reshape(-1, 1)).flatten()
    
    fig, ax = plt.subplots()
    ax.plot(data, label='Original')
    ax.plot(normalized_data, label='Normalized')
    ax.set_title('Original vs. Normalized Data')
    ax.legend()
    
    return data, normalized_data, ax