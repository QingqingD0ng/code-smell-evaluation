from datetime import datetime, timedelta
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(days_in_past=7, random_seed=0):
    random.seed(random_seed)
    activities = ["Running", "Swimming", "Cycling", "Yoga", "Weight Training"]
    date = datetime.now() - timedelta(days=days_in_past)
    dates = [date + timedelta(days=i) for i in range(days_in_past)]
    data = []
    for activity in activities:
        for date in dates:
            duration = random.randint(0, 120)
            data.append([date.strftime("%Y-%m-%d"), activity, duration])
    df = pd.DataFrame(data, columns=["Date", "Activity", "Duration"])
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    sns.lineplot(data=df, x=df.index, y="Duration", hue="Activity")
    ax = plt.gca()
    return ax, df