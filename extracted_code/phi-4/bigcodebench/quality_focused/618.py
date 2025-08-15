from random import randint
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TEAMS = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E']
PENALTY_COST = 1000

def task_func(goals, penalties):
    data = {
        'Team': [],
        'Goals': [],
        'Penalty Cost': []
    }
    
    for team in TEAMS:
        team_goals = randint(0, goals)
        team_penalties = randint(0, penalties)
        penalty_cost = team_penalties * PENALTY_COST
        data['Team'].append(team)
        data['Goals'].append(team_goals)
        data['Penalty Cost'].append(penalty_cost)
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 5))
    
    ax_goals = plt.subplot(1, 2, 1)
    sns.barplot(x='Team', y='Goals', data=df, ax=ax_goals)
    ax_goals.set_title('Goals per Team')
    
    ax_penalty = plt.subplot(1, 2, 2)
    sns.barplot(x='Team', y='Penalty Cost', data=df, ax=ax_penalty, palette='coolwarm')
    ax_penalty.set_title('Penalty Cost per Team')
    
    plt.tight_layout()
    plt.show()
    
    return df, [ax_goals, ax_penalty]

# Example usage:
# df, plots = task_func(5, 3)