import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def task_func(df: pd.DataFrame) -> tuple:
    fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [0.2, 0.8]})
    
    sns.boxplot(y=df['closing_price'], ax=ax_box)
    ax_box.set_title('Box Plot of Closing Prices')
    ax_box.set_ylabel('')

    sns.histplot(df['closing_price'], kde=False, ax=ax_hist, color='skyblue', bins=10)
    ax_hist.set_title('Histogram of Closing Prices')
    ax_hist.set_xlabel('Closing Price')
    ax_hist.set_ylabel('Frequency')

    plt.tight_layout()
    return ax_box, ax_hist