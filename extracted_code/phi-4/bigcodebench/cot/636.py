import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLUMNS = ['Column1', 'Column2', 'Column3', 'Column4', 'Column5']

def task_func(rows):
    data = np.random.randint(0, 10, size=(rows, len(COLUMNS)))
    df = pd.DataFrame(data, columns=COLUMNS)
    non_zero_counts = (df!= 0).sum()
    
    fig, ax = plt.subplots()
    non_zero_counts.plot(kind='bar', ax=ax)
    ax.set_title('Non-Zero Value Counts')
    
    return df, ax