import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLUMNS = ['A', 'B', 'C', 'D', 'E']

def task_func(a, b):
    data = np.random.rand(len(a), len(COLUMNS))
    df = pd.DataFrame(data, index=a, columns=b)
    fig, ax = plt.subplots()
    df.plot(kind='bar', ax=ax)
    return ax