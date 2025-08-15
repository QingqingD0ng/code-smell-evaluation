import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

def task_func(data):
    df = pd.DataFrame(data, columns=['Item', 'Count', 'Weight'])
    df['Normalized Count'] = zscore(df['Count'])
    scaler = MinMaxScaler()
    df['Normalized Weight'] = scaler.fit_transform(df[['Weight']]).flatten()
    return df[['Item', 'Normalized Count', 'Normalized Weight']]