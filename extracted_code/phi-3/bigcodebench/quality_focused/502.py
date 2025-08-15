from datetime import datetime, timedelta
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(days_in_past=7, random_seed=0):
    random.seed(random_seed)
    today = datetime.now()
    start_date = today - timedelta(days=days_in_past)
    activities = ["Running", "Swimming", "Cycling", "Yoga", "Weight Training"]
    data = {
        "Date": [],
        "Activity": [],
        "Duration": []
    }
    for _ in range(days_in_past * len(activities)):
        date = start_date + timedelta(days=random.randint(0, days_in_past))
        activity = random.choice(activities)
        duration = random.randint(0, 120)
        data["Date"].append(date)
        data["Activity"].append(activity)
        data["Duration"].append(duration)

    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    ax = sns.lineplot(data=df, x="Date", y="Duration", hue="Activity")
    ax.set_title(f"Daily Activity Durations for the Past {days_in_past} Days")
    ax.set(xlabel="Date", ylabel="Duration (minutes)")

    return ax, df

# Example usage:
ax, df = task_func(7, random_seed=42)