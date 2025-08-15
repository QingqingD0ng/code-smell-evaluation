import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns


def task_func(df: pd.DataFrame) -> tuple:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    
    sns.boxplot(x=df['closing_price'], ax=axes[0])
    axes[0].set_title('Box Plot of Closing Prices')
    
    sns.histplot(df['closing_price'], kde=True, ax=axes[1])
    axes[1].set_title('Histogram of Closing Prices')
    
    plt.tight_layout()
    
    return axes