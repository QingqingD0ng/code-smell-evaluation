import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLUMNS = ['A', 'B', 'C', 'D', 'E']

def task_func(a, b):
    num_rows = len(a)
    num_cols = min(len(b), len(COLUMNS))
    data = np.random.rand(num_rows, num_cols)
    df = pd.DataFrame(data, index=a, columns=COLUMNS[:num_cols])
    ax = df.plot(kind='bar')
    plt.show()
    return ax