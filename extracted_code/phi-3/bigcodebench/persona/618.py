import random

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns


TEAMS = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E']

PENALTY_COST = 1000


def task_func(goals, penalties):

    # Generate random goals and penalties for each team

    data = {

        'Team': [team for team in TEAMS for _ in range(2)],  # Assuming 2 matches per team

        'Goals': [random.randint(0, goals) for _ in range(2 * len(TEAMS))],

        'Penalties': [random.randint(0, penalties) for _ in range(2 * len(TEAMS))],

        'Penalty Cost': [PENALTY_COST * penalties for _ in range(2 * len(TEAMS))],

    }


    # Create a DataFrame

    df = pd.DataFrame(data)


    # Seaborn plots

    goals_plot = sns.barplot(x='Team', y='Goals', data=df)

    penalty_cost_plot = sns.barplot(x='Team', y='Penalty Cost', data=df)


    return df, [goals_plot, penalty_cost_plot]


# Example usage

df, plots = task_func(5, 3)

for plot in plots:

    plt.show()