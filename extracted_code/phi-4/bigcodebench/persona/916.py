import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def task_func(df: pd.DataFrame) -> tuple:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Box plot
    sns.boxplot(x=df['closing_price'], ax=axes[0])
    axes[0].set_title('Box Plot of Closing Prices')

    # Histogram
    sns.histplot(df['closing_price'], kde=False, ax=axes[1])
    axes[1].set_title('Histogram of Closing Prices')

    plt.tight_layout()
    return axes[0], axes[1]