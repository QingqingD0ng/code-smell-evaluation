from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

def task_func(activities):
    if not all(isinstance(activity, datetime) for activity in activities):
        raise TypeError("All activities must be datetime objects.")

    days_count = defaultdict(int)
    for activity in activities:
        days_count[activity.strftime("%A")] += 1

    days = list(days_count.keys())
    counts = list(days_count.values())

    plt.figure(figsize=(10, 6))
    ax = plt.bar(days, counts)
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Activities')
    plt.title('Weekly Activity')
    plt.show()

    return ax