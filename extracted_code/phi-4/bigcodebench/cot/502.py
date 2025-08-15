from datetime import datetime, timedelta
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt

def task_func(days_in_past=7, random_seed=0):
    random.seed(random_seed)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_in_past)
    
    dates = [start_date + timedelta(days=i) for i in range(days_in_past)]
    activities = ["Running", "Swimming", "Cycling", "Yoga", "Weight Training"]
    
    data = {
        "Date": [],
        "Activity": [],
        "Duration": []
    }
    
    for date in dates:
        for activity in activities:
            duration = random.randint(0, 120)
            data["Date"].append(date)
            data["Activity"].append(activity)
            data["Duration"].append(duration)
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=df, x="Date", y="Duration", hue="Activity", marker="o")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return ax, df