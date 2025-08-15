import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import pandas as pd

def task_func(df, z_threshold=2):
    if 'closing_price' not in df.columns:
        raise ValueError("DataFrame must contain a 'closing_price' column.")

    df['z_score'] = zscore(df['closing_price'])
    outliers = df[np.abs(df['z_score']) > z_threshold]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['closing_price'], label='Closing Price')
    ax.scatter(outliers.index, outliers['closing_price'], color='red', label='Outliers')
    ax.set_xlabel('Index')
    ax.set_ylabel('Closing Price')
    ax.set_title('Outliers in Closing Prices')
    ax.legend()
    
    return outliers, ax

# Example usage:
# import pandas as pd
# df1 = pd.DataFrame({
#     'closing_price': [100, 101, 102, 103, 104, 150]
# })
# outliers1, plot1 = task_func(df1)

# df2 = pd.DataFrame({
#     'closing_price': [10, 20, 30, 40, 50, 100]
# })
# outliers2, plot2 = task_func(df2, z_threshold=1.5)