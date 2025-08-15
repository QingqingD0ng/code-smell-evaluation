import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

def task_func(original):
    # Unzip the list of tuples into a numpy array
    arr = np.array([item[1] for item in original])
    
    # Normalize the array
    norm_arr = preprocessing.minmax_scale(arr)
    
    # Create a figure and axis for plotting
    fig, ax = plt.subplots()
    
    # Plot the original and normalized arrays
    ax.plot(arr, label='Original')
    ax.plot(norm_arr, label='Normalized')
    
    # Set the title and legend
    ax.set_title('Original vs. Normalized Data')
    ax.legend()
    
    # Return the original array, normalized array, and axes object
    return arr, norm_arr, ax