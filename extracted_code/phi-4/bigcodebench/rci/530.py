import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(df: pd.DataFrame):
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if not {'name', 'age'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'name' and 'age' columns")
    
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(0).astype(int)
    df['age'] = np.floor(df['age']).astype(int)
    
    if df['age'].min() < 0:
        raise ValueError("Age must not be negative")
    
    duplicate_names = df[df.duplicated('name', keep=False)]
    
    if duplicate_names.empty:
        return Counter(), False
    
    age_distribution = Counter(duplicate_names['age'])
    
    plt.figure()
    bin_start = duplicate_names['age'].min() - 0.5
    bin_end = duplicate_names['age'].max() + 1.5
    ax = sns.histplot(duplicate_names['age'], bins=np.arange(bin_start, bin_end), kde=False)
    ax.set(xlabel='Age', ylabel='Count')
    
    return age_distribution, ax