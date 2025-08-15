import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

def task_func(df: pd.DataFrame) -> (Counter, plt.Axes):
    if df.empty:
        raise ValueError("DataFrame is empty")
    if any(df['age'] < 0):
        raise ValueError("Age cannot be negative")

    df['age'] = df['age'].apply(lambda x: int(np.floor(x)))

    duplicates = df.duplicated(subset='name', keep=False)
    if duplicates.sum() == 0:
        return Counter(), None

    duplicates_df = df[duplicates]
    name_age_counter = Counter(zip(duplicates_df['name'], duplicates_df['age']))

    plt.figure(figsize=(10, 6))
    sns.histplot(duplicates_df['age'], bins=np.arange(min(duplicates_df['age']), max(duplicates_df['age']) + 1) - 0.5, kde=False, stat='count')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Age distribution among duplicate names')
    plt.xticks(range(int(min(duplicates_df['age'])), int(max(duplicates_df['age']))))

    return name_age_counter, plt.gca()