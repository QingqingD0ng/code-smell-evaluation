import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLUMNS = ['A', 'B', 'C', 'D', 'E']

def task_func(a, b):
    num_rows = len(a)
    num_columns = min(len(b), len(COLUMNS))
    
    data = np.random.rand(num_rows, num_columns)
    df = pd.DataFrame(data, index=a, columns=COLUMNS[:num_columns])
    
    ax = df.plot(kind='bar', figsize=(10, 6))
    plt.ylabel('Values')
    plt.title('Bar Chart of Random Data')
    
    return ax