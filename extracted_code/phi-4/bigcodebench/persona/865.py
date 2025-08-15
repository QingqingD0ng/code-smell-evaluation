import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

def task_func(data):
    df = pd.DataFrame(data, columns=['Item', 'Count', 'Weight'])
    
    # Normalize counts using z-score normalization
    df['Normalized Count'] = zscore(df['Count'])
    
    # Normalize weights using min-max scaling
    scaler = MinMaxScaler()
    df['Normalized Weight'] = scaler.fit_transform(df[['Weight']])
    
    # Select relevant columns for the final DataFrame
    result_df = df[['Item', 'Normalized Count', 'Normalized Weight']]
    
    return result_df