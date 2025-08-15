from datetime import datetime

from collections import defaultdict

import matplotlib.pyplot as plt


def task_func(activities):

    if not all(isinstance(activity, datetime) for activity in activities):

        raise TypeError("All activities must be datetime objects")


    days = defaultdict(int)

    for activity in activities:

        day_of_week = activity.strftime('%A')

        days[day_of_week] += 1


    days_sorted = sorted(days.items())

    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    counts = [count for day, count in days_sorted if day in days_of_week]


    plt.bar(days_of_week, counts)

    plt.xlabel('Day of the Week')

    plt.ylabel('Number of Activities')

    plt.title('Weekly Activity')

    return plt.gca()