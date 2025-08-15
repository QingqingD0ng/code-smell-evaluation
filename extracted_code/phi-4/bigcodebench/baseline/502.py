from datetime import datetime, timedelta
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(days_in_past=7, random_seed=0):
    random.seed(random_seed)
    activities = ["Running", "Swimming", "Cycling", "Yoga", "Weight Training"]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_in_past)
    
    data = []
    for i in range(days_in_past):
        date = start_date + timedelta(days=i)
        for activity in activities:
            duration = random.randint(0, 120)
            data.append([date, activity, duration])
    
    df = pd.DataFrame(data, columns=["Date", "Activity", "Duration"])
    
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df, x="Date", y="Duration", hue="Activity", marker="o")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return ax, df

# Example usage
ax, df = task_func(7, random_seed=42)
plt.show()