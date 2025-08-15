import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLUMNS = ['Column1', 'Column2', 'Column3', 'Column4', 'Column5']

def task_func(rows):
    df = pd.DataFrame(np.random.randint(0, 10, size=(rows, len(COLUMNS))), columns=COLUMNS)
    non_zero_counts = df.notnull().sum()
    fig, ax = plt.subplots()
    non_zero_counts.plot(kind='bar', ax=ax)
    ax.set_title('Non-Zero Value Counts')
    plt.close(fig)
    
    return df, ax