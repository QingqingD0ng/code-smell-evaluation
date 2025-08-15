import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(df: pd.DataFrame) -> (Counter, plt.Axes):
    if df.empty:
        raise ValueError("DataFrame is empty")
    if (df['age'] < 0).any():
        raise ValueError("Age cannot be negative")

    df['age'] = df['age'].apply(np.floor).astype(int)
    duplicates = df[df.duplicated(subset=['name'], keep=False)]

    if duplicates.empty:
        return Counter(), None

    age_distribution = Counter(duplicates['age'])

    min_age, max_age = duplicates['age'].min(), duplicates['age'].max()
    bins = np.arange(min_age, max_age + 1, 1) - 0.5

    ax = sns.histplot(duplicates, x='age', bins=bins, kde=False)
    ax.set_title('Age distribution among duplicate names')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')

    return age_distribution, ax