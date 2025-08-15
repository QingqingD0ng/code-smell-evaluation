from datetime import datetime, timedelta
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(days_in_past=7, random_seed=0):
    random.seed(random_seed)
    activities = ["Running", "Swimming", "Cycling", "Yoga", "Weight Training"]
    end_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days_in_past)

    data = []
    for single_date in (start_date + timedelta(n) for n in range(days_in_past)):
        for activity in activities:
            duration = random.randint(0, 120)
            data.append({"Date": single_date, "Activity": activity, "Duration": duration})

    df = pd.DataFrame(data)
    ax = sns.lineplot(data=df, x="Date", y="Duration", hue="Activity", marker="o")
    plt.xticks(rotation=45)
    plt.tight_layout()

    return ax, df