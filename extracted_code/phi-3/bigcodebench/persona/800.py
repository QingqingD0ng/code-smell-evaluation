import csv

from collections import Counter


CSV_FILE_PATH ='match_data.csv'


def update_csv_with_goals_and_penalties(goals, penalties, csv_file_path=CSV_FILE_PATH):
    # Read the current CSV file data
    with open(csv_file_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip header
        current_goals = {row[0]: int(row[1]) for row in csv_reader}
        current_penalties = {row[0]: int(row[2]) for row in csv_reader}

    # Update the current values with the new goals and penalties
    for team, goal in goals.items():
        current_goals[team] += goal
    for team, penalty in penalties.items():
        current_penalties[team] += penalty

    # Write the updated CSV file
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['team', 'goals', 'penalties'])  # Write header
        for team in current_goals:
            csv_writer.writerow([team, current_goals[team], current_penalties[team]])

    return Counter(current_goals) + Counter(current_penalties)