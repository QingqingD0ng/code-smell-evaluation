import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def task_func(df: pd.DataFrame) -> tuple:
    if 'closing_price' not in df.columns:
        raise ValueError("DataFrame must contain a 'closing_price' column")

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    sns.boxplot(data=df, x='closing_price', ax=axes[0])
    axes[0].set_title('Box Plot of Closing Prices')

    sns.histplot(data=df, x='closing_price', kde=True, ax=axes[1])
    axes[1].set_title('Histogram of Closing Prices')

    plt.tight_layout()

    return axes[0], axes[1]