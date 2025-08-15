import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta

def generate_activity_data(days, seed=0):
    np.random.seed(seed)
    activities = ["Running", "Swimming", "Cycling", "Yoga", "Weight Training"]
    start_date = pd.Timestamp.now() - timedelta(days=days)
    dates = [start_date + timedelta(days=x) for x in range(days)]
    data = {
        "Date": np.repeat(dates, len(activities)),
        "Activity": np.tile(activities, days),
        "Duration": np.random.randint(0, 121, size=len(dates) * len(activities))
    }
    return pd.DataFrame(data)

def plot_activity_data(df):
    df.set_index("Date", inplace=True)
    ax = sns.lineplot(data=df, x=df.index, y="Duration", hue="Activity", estimator=None)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return ax, df

def main(days_in_past=7, random_seed=0):
    df = generate_activity_data(days_in_past, random_seed)
    ax, df = plot_activity_data(df)
    return ax, df

ax, df = main(7, random_seed=42)