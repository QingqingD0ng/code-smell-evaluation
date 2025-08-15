from random import randint
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Constants
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
        penalty_cost = team_penalties * PENALTY_COST
        
        data['Team'].append(team)
        data['Goals'].append(team_goals)
        data['Penalties'].append(team_penalties)
        data['Penalty Cost'].append(penalty_cost)
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    goals_plot = sns.barplot(x='Team', y='Goals', data=df)
    goals_plot.set_title('Goals per Team')
    
    plt.subplot(1, 2, 2)
    penalty_cost_plot = sns.barplot(x='Team', y='Penalty Cost', data=df, palette='coolwarm')
    penalty_cost_plot.set_title('Penalty Cost per Team')
    
    plt.tight_layout()
    
    return df, [goals_plot, penalty_cost_plot]

# Example usage
df, plots = task_func(5, 3)
plt.show()