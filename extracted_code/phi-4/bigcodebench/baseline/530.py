import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(df: pd.DataFrame):
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if ('name' not in df.columns) or ('age' not in df.columns):
        raise ValueError("DataFrame must contain 'name' and 'age' columns")
    
    df['age'] = np.floor(df['age']).astype(int)
    
    if df['age'].min() < 0:
        raise ValueError("Age must not be negative")
    
    duplicate_names = df[df.duplicated('name', keep=False)]
    
    if duplicate_names.empty:
        return Counter(), None
    
    age_distribution = Counter(duplicate_names['age'])
    
    plt.figure()
    ax = sns.histplot(duplicate_names['age'], bins=np.arange(duplicate_names['age'].min() - 0.5, 
                                                              duplicate_names['age'].max() + 1.5), 
                      kde=False)
    ax.set(xlabel='Age', ylabel='Count')
    
    return age_distribution, ax