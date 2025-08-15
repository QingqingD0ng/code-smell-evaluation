import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(df):
    if df.empty:
        raise ValueError("DataFrame is empty")
    if (df['age'] < 0).any():
        raise ValueError("Age cannot be negative")
    
    df['age'] = df['age'].apply(np.floor)
    duplicates = df.duplicated(subset='name', keep=False)
    
    if not duplicates.any():
        return Counter(), None
    
    duplicate_names = df[duplicates]['name']
    age_distribution = Counter(df[duplicates]['age'])
    
    min_age, max_age = df[duplicates]['age'].min(), df[duplicates]['age'].max()
    bins = np.arange(min_age - 0.5, max_age + 1.5, 1)
    
    ax = sns.histplot(df[duplicates], x='age', bins=bins, kde=False)
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    ax.set_title('Age distribution among duplicate names')
    
    return age_distribution, ax