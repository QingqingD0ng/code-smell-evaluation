import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLUMNS = ['A', 'B', 'C', 'D', 'E']

def task_func(a, b):
    num_rows = len(a)
    num_cols = len(b)
    df = pd.DataFrame(np.random.rand(num_rows, num_cols), index=a, columns=b)
    ax = df.plot(kind='bar')
    return ax

# Example usage:
ax = task_func([1, 2, 3], ['A', 'B', 'C', 'D', 'E'])
plt.show()