import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def task_func(df: pd.DataFrame) -> tuple:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    sns.boxplot(data=df, x='closing_price', ax=ax1)
    ax1.set_title('Box Plot of Closing Prices')

    sns.histplot(data=df, x='closing_price', ax=ax2, kde=True)
    ax2.set_title('Histogram of Closing Prices')

    plt.tight_layout()
    return ax1, ax2