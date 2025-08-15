import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLUMNS = ['A', 'B', 'C', 'D', 'E']

def task_func(a, b):
    num_rows = len(a)
    num_columns = min(len(b), len(COLUMNS))
    column_names = COLUMNS[:num_columns]

    # Generate random data
    data = np.random.rand(num_rows, num_columns)

    # Create DataFrame
    df = pd.DataFrame(data, index=a, columns=column_names)

    # Plotting
    ax = df.plot(kind='bar')
    plt.show()
    
    return ax