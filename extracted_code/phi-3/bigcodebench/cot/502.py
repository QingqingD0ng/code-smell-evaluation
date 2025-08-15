from datetime import datetime, timedelta
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(days_in_past=7, random_seed=0):
    random.seed(random_seed)
    today = datetime.now()
    past_date = today - timedelta(days=days_in_past)
    activities = ["Running", "Swimming", "Cycling", "Yoga", "Weight Training"]
    dates = [past_date + timedelta(days=i) for i in range(days_in_past)]
    df = pd.DataFrame({
        "Date": [date.strftime("%Y-%m-%d") for date in dates],
        "Activity": [random.choice(activities) for _ in range(days_in_past)],
        "Duration": [random.randint(0, 120) for _ in range(days_in_past)]
    })
    ax = sns.lineplot(data=df, x="Date", y="Duration", hue="Activity")
    plt.show()

task_func(7, random_seed=42)  # Example usage