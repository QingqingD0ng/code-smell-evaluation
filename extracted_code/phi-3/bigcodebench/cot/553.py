import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLUMNS = ['A', 'B', 'C', 'D', 'E']

def generate_dataframe_and_plot(a, b):
    num_rows = len(a)
    index = pd.RangeIndex(start=0, stop=num_rows, step=1)
    data = np.random.rand(num_rows, len(b))
    df = pd.DataFrame(data, index=index, columns=b)
    
    ax = df.plot(kind='bar')
    plt.show()
    
    return ax

ax = generate_dataframe_and_plot([1, 2, 3], ['A', 'B', 'C', 'D', 'E'])