import pandas as pd

import numpy as np

from scipy.stats import zscore

from sklearn.preprocessing import MinMaxScaler


def task_func(data):
    df = pd.DataFrame(data, columns=['Item', 'Count', 'Weight'])
    df['Normalized Count'] = zscore(df['Count'])
    scaler = MinMaxScaler()
    df['Normalized Weight'] = scaler.fit_transform(df[['Weight']])
    return df


# Example usage:

data = [('A', 100, 0.5), ('B', 200, 0.6), ('C', 150, 0.7)]
print(task_func(data))

data = [('Andrew', 5743, 0.925), ('Elizabeth', 4655, 1.0875), ('Susan', 4716, 0.65), ('Christopher', 2100, 0.05),('Timothy', 3943, 0.175)]
print(task_func(data))