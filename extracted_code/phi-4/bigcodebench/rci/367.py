from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import calendar

def task_func(activities):
    if not all(isinstance(activity, datetime) for activity in activities):
        raise TypeError("All items in 'activities' must be datetime objects. Example: datetime(2023, 10, 25).")
    
    days_of_week = list(calendar.day_name)
    activity_counts = defaultdict(int)
    
    for activity in activities:
        day_of_week = activity.strftime("%A")
        activity_counts[day_of_week] += 1
    
    counts = [activity_counts[day] for day in days_of_week]
    
    fig, ax = plt.subplots()
    ax.bar(days_of_week, counts)
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Number of Activities')
    ax.set_title('Weekly Activity')
    
    return ax