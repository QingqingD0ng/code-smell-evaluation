from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

def task_func(activities):
    if not all(isinstance(activity, datetime) for activity in activities):
        raise TypeError("All activities must be datetime objects.")

    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    activity_count = defaultdict(int)

    for activity in activities:
        day_index = activity.weekday()
        activity_count[days_of_week[day_index]] += 1

    fig, ax = plt.subplots()
    ax.bar(activity_count.keys(), activity_count.values())

    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Number of Activities')
    ax.set_title('Weekly Activity')

    plt.xticks(rotation=45)
    plt.tight_layout()

    return ax