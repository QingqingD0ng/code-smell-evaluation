import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

def normalize_counts(counts):
    return zscore(counts)

def normalize_weights(weights):
    scaler = MinMaxScaler()
    return scaler.fit_transform(weights.values.reshape(-1, 1)).flatten()

def task_func(data):
    df = pd.DataFrame(data, columns=['Item', 'Count', 'Weight'])
    df['Normalized Count'] = normalize_counts(df['Count'])
    df['Normalized Weight'] = normalize_weights(df['Weight'])
    return df[['Item', 'Normalized Count', 'Normalized Weight']]