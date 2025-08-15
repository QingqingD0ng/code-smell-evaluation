import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLUMNS = ['A', 'B', 'C', 'D', 'E']
MAX_COLUMNS = len(COLUMNS)

def generate_bar_chart(row_indices, _):
    num_rows = len(row_indices)
    num_cols = min(num_rows, MAX_COLUMNS)
    df = pd.DataFrame(np.random.rand(num_rows, num_cols), index=row_indices, columns=COLUMNS[:num_cols])
    ax = df.plot(kind='bar')
    return ax