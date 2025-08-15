import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler

def task_func(data):
    elements, counts, weights = zip(*data)
    counts_zscore = zscore(np.array(counts))
    weights_scaler = MinMaxScaler()
    weights_scaled = weights_scaler.fit_transform(np.array(weights).reshape(-1, 1)).flatten()
    
    return pd.DataFrame({'Item': elements, 'Normalized Count': counts_zscore, 'Normalized Weight': weights_scaled})