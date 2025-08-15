import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import List, Any

def task_func(activities: List[datetime.datetime]) -> plt.Axes:
    """
    Return a bar chart of the number of activities performed on each day of the week based on the provided list of activities.
    If the activities are not datetime objects, raise a TypeError.
    
    :param activities: A list of datetime objects representing when each activity occurred.
    :return: Axes object representing the bar chart.
    """
    if not all(isinstance(activity, datetime.datetime) for activity in activities):
        raise TypeError("All activities must be datetime objects")

    day_counts = defaultdict(int)
    for activity in activities:
        day_counts[activity.strftime("%A")] += 1

    # Create the bar chart
    days = list(day_counts.keys())
    counts = list(day_counts.values())
    fig, ax = plt.subplots()
    ax.bar(days, counts, color='skyblue')  # Customized color for better visualization

    # Customize the chart
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Number of Activities')
    ax.set_title('Weekly Activity')
    ax.set_xticklabels(days, rotation=45, ha="right")  # Rotate x-axis labels for better readability

    return ax

# Example usage
# ax = task_func([datetime.datetime(2023, 10, 25), datetime.datetime(2023, 10, 26)])
# print(type(ax))