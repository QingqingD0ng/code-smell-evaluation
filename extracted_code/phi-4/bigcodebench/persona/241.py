import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def task_func(original):
    # Extract the numeric values from the original list of tuples
    data = np.array([item[1] for item in original])
    
    # Normalize the data using MinMaxScaler
    scaler = preprocessing.MinMaxScaler()
    norm_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # Plotting the original and normalized data
    fig, ax = plt.subplots()
    ax.plot(data, label='Original', marker='o')
    ax.plot(norm_data, label='Normalized', marker='x')
    ax.set_title('Original vs. Normalized Data')
    ax.legend()
    
    return data, norm_data, ax