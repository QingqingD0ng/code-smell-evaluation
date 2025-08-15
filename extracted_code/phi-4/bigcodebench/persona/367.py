from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

def task_func(activities):
    # Check if all items in activities are datetime objects
    if not all(isinstance(activity, datetime) for activity in activities):
        raise TypeError("All activities must be datetime objects.")

    # Dictionary to count activities for each day of the week
    day_count = defaultdict(int)

    # Loop through each activity and count occurrences for each day
    for activity in activities:
        day_of_week = activity.strftime('%A')
        day_count[day_of_week] += 1

    # Prepare data for plotting
    days = list(day_count.keys())
    counts = list(day_count.values())

    # Create bar chart
    fig, ax = plt.subplots()
    ax.bar(days, counts, color='skyblue')

    # Set labels and title
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Number of Activities')
    ax.set_title('Weekly Activity')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show()

    return ax