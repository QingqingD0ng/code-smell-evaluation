import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(df: pd.DataFrame):
    if df.empty:
        raise ValueError("DataFrame is empty")

    if not all(df['age'] >= 0):
        raise ValueError("Age must not be negative")

    df['age'] = df['age'].apply(np.floor).astype(int)

    duplicates = df[df.duplicated('name', keep=False)]
    if duplicates.empty:
        return Counter(), None

    age_distribution = Counter(duplicates['age'])

    min_age, max_age = min(age_distribution), max(age_distribution)
    bins = np.arange(min_age - 0.5, max_age + 1.5, 1)

    plt.figure()
    ax = sns.histplot(duplicates['age'], bins=bins, kde=False)
    ax.set(xlabel='Age', ylabel='Count')
    ax.set(xticks=range(min_age, max_age + 1))

    return age_distribution, ax