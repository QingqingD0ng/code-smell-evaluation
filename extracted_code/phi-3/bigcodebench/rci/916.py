import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def task_func(df: pd.DataFrame) -> tuple:
    """Visualize stock closing price distribution with box plot and histogram."""
    if 'closing_price' not in df.columns:
        raise ValueError("DataFrame must contain 'closing_price' column.")

    # Set up the matplotlib figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Box plot
    sns.boxplot(data=df, ax=axes[0])
    axes[0].set_title('Box Plot of Closing Prices')

    # Histogram
    sns.histplot(df['closing_price'], kde=False, ax=axes[1])
    axes[1].set_title('Histogram of Closing Prices')

    plt.tight_layout()
    plt.show()

    return axes

# Example
df_example = pd.DataFrame({
    'closing_price': [100, 101, 102, 103, 104, 150]
})
axes_example = task_func(df_example)
# Accessing titles of the axes
print(axes_example[0].get_title())
print(axes_example[1].get_title())