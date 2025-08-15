import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def task_func(df: pd.DataFrame) -> tuple:
    if 'closing_price' not in df.columns:
        raise ValueError("DataFrame must contain a 'closing_price' column.")

    # Set up the matplotlib figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Create a Box Plot
    sns.boxplot(x=df['closing_price'], ax=axes[0])
    axes[0].set_title('Box Plot of Closing Prices')

    # Create a Histogram
    sns.histplot(df['closing_price'], kde=False, ax=axes[1])
    axes[1].set_title('Histogram of Closing Prices')

    plt.tight_layout()
    plt.show()

    return axes

# Example usage:
# df = pd.DataFrame({
#     'closing_price': [100, 101, 102, 103, 104, 150]
# })
# boxplot_ax, histplot_ax = task_func(df)
# print(boxplot_ax.get_title())
# print(histplot_ax.get_title())