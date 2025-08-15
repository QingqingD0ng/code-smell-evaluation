import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

def task_func(activities):
    if not all(isinstance(activity, datetime.datetime) for activity in activities):
        raise TypeError("All activities must be datetime objects")
    
    day_counts = defaultdict(int)
    for activity in activities:
        day_counts[activity.strftime("%A")] += 1
    
    days = list(day_counts.keys())
    counts = list(day_counts.values())
    
    fig, ax = plt.subplots()
    ax.bar(days, counts)
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Number of Activities')
    ax.set_title('Weekly Activity')
    
    return ax