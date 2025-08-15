import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(df: pd.DataFrame):
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if 'name' not in df or 'age' not in df:
        raise ValueError("DataFrame must contain 'name' and 'age' columns")
    
    if (df['age'] < 0).any():
        raise ValueError("Age must not be negative")
    
    df['age'] = df['age'].apply(np.floor).astype(int)
    
    name_counts = df['name'].value_counts()
    duplicate_names = name_counts[name_counts > 1].index
    duplicates_df = df[df['name'].isin(duplicate_names)]
    
    if duplicates_df.empty:
        return Counter(), None
    
    age_distribution = Counter(duplicates_df['age'])
    
    plt.figure(figsize=(8, 6))
    ax = sns.histplot(duplicates_df['age'], bins=np.arange(min(duplicates_df['age']) - 0.5, 
                                                            max(duplicates_df['age']) + 1.5, 1), 
                      kde=False, discrete=True)
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    
    return age_distribution, ax