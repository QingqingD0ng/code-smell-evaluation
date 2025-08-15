import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(df: pd.DataFrame):
    if df.empty:
        raise ValueError("The DataFrame is empty.")
    
    if (df['age'] < 0).any():
        raise ValueError("Age must not be negative.")
    
    df['age'] = np.floor(df['age']).astype(int)
    
    duplicates = df[df.duplicated('name', keep=False)]
    if duplicates.empty:
        return Counter(), None
    
    age_distribution = Counter(duplicates['age'])
    
    plt.figure()
    ax = sns.histplot(duplicates['age'], bins=np.arange(duplicates['age'].min() - 0.5, duplicates['age'].max() + 1.5, 1), kde=False)
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    
    return age_distribution, ax