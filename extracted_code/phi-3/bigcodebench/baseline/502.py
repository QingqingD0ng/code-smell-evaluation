import datetime
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(days_in_past=7, random_seed=0):
    random.seed(random_seed)
    activities = ["Running", "Swimming", "Cycling", "Yoga", "Weight Training"]
    start_date = datetime.datetime.now() - datetime.timedelta(days=days_in_past)
    dates = [start_date + datetime.timedelta(days=x) for x in range(days_in_past)]
    data = {
        "Date": [],
        "Activity": [],
        "Duration": []
    }
    for date in dates:
        for activity in activities:
            data["Date"].append(date)
            data["Activity"].append(activity)
            data["Duration"].append(random.randint(0, 120))
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    ax = sns.lineplot(data=df, x="Date", y="Duration", hue="Activity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return ax, df

ax, df = task_func(7, random_seed=42)