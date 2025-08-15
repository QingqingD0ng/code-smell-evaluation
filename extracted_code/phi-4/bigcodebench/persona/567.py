import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def task_func(data):
    # Convert the string into a list of integers
    values = list(map(int, data.split('-')))
    
    # Create a DataFrame
    df = pd.DataFrame(values, columns=['Value'])
    
    # Calculate histogram bins
    bins = np.arange(df['Value'].min(), df['Value'].max() + 2) - 0.5
    
    # Plot the histogram
    ax = df['Value'].plot(kind='hist', bins=bins, edgecolor='black')
    
    # Set labels and title
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Values')
    
    # Return the Axes object
    return ax