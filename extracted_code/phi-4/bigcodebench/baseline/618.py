from random import randint
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TEAMS = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E']
PENALTY_COST = 1000  # in dollars

def task_func(goals, penalties):
    data = {
        'Team': [],
        'Goals': [],
        'Penalties': [],
        'Penalty Cost': []
    }

    for team in TEAMS:
        team_goals = randint(0, goals)
        team_penalties = randint(0, penalties)
        team_penalty_cost = team_penalties * PENALTY_COST
        
        data['Team'].append(team)
        data['Goals'].append(team_goals)
        data['Penalties'].append(team_penalties)
        data['Penalty Cost'].append(team_penalty_cost)

    df = pd.DataFrame(data)

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(x='Team', y='Goals', data=df, ax=ax[0], palette='viridis')
    ax[0].set_title('Goals by Team')
    ax[0].set_ylabel('Goals')

    sns.barplot(x='Team', y='Penalty Cost', data=df, ax=ax[1], palette='magma')
    ax[1].set_title('Penalty Cost by Team')
    ax[1].set_ylabel('Penalty Cost ($)')

    plt.tight_layout()
    plt.show()

    return df, ax