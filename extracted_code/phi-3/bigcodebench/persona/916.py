import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns


def task_func(df: pd.DataFrame) -> tuple:

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    sns.boxplot(x=df['closing_price'], ax=ax[0], title='Box Plot of Closing Prices')

    sns.histplot(df['closing_price'], kde=True, ax=ax[1], stat='density', title='Histogram of Closing Prices')

    return ax[0], ax[1]