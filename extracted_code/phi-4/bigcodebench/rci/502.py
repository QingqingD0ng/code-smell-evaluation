from datetime import datetime, timedelta
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt

ACTIVITIES = ["Running", "Swimming", "Cycling", "Yoga", "Weight Training"]
DURATION_MIN = 0
DURATION_MAX = 120

def generate_activity_data(days_in_past, random_seed):
    random.seed(random_seed)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_in_past)
    
    data = []
    for i in range(days_in_past):
        date = start_date + timedelta(days=i)
        for activity in ACTIVITIES:
            duration = random.randint(DURATION_MIN, DURATION_MAX)
            data.append([date, activity, duration])
    
    return pd.DataFrame(data, columns=["Date", "Activity", "Duration"])

def plot_activity_data(df, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    ax = sns.lineplot(data=df, x="Date", y="Duration", hue="Activity", marker="o")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return ax

def task_func(days_in_past=7, random_seed=0):
    if days_in_past < 0:
        raise ValueError("days_in_past must be non-negative")
    
    df = generate_activity_data(days_in_past, random_seed)
    ax = plot_activity_data(df)
    return ax, df

# Example usage
ax, df = task_func(7, random_seed=42)
plt.show()