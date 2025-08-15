import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLUMNS = ['Column1', 'Column2', 'Column3', 'Column4', 'Column5']

def task_func(rows):
    df = pd.DataFrame(np.random.randint(0, 10, size=(rows, len(COLUMNS))), columns=COLUMNS)
    non_zero_counts = (df!= 0).sum()
    
    ax = non_zero_counts.plot(kind='bar', figsize=(8, 6))
    ax.set_title('Non-Zero Value Counts')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Count')
    
    return df, ax

# Example usage
# df, ax = task_func(10)
# plt.show()