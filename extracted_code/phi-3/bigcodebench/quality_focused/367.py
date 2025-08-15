from datetime import datetime

from collections import defaultdict

import matplotlib.pyplot as plt


def task_func(activities):
    if not all(isinstance(activity, datetime) for activity in activities):
        raise TypeError("All activities must be datetime objects")

    activity_count_by_day = defaultdict(int)
    for activity in activities:
        activity_day = activity.strftime('%A')
        activity_count_by_day[activity_day] += 1

    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    activities_count = list(activity_count_by_day.values())

    plt.bar(days_of_week, activities_count, color='skyblue')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Activities')
    plt.title('Weekly Activity')
    return plt.gca()


# Example usage:

# ax = task_func([datetime(2023, 10, 25), datetime(2023, 10, 26)])
# print(type(ax))