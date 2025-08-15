import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(data):
    # Convert string data to a list of integers
    values = np.fromstring(data, sep='-', dtype=int)
    
    # Create a pandas Series from the list of values
    series = pd.Series(values)
    
    # Calculate bin edges with np.arange
    bins = np.arange(series.min(), series.max() + 2) - 0.5
    
    # Plot histogram
    ax = series.plot.hist(bins=bins, title='Histogram of Values', y='Frequency')
    
    # Set x-axis label
    ax.set_xlabel('Value')
    
    return ax