from random import randint
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TEAMS = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E']
PENALTY_COST = 1000

def task_func(goals, penalties):
    data = {'Team': [], 'Goals': [], 'Penalty Cost': []}
    
    for team in TEAMS:
        team_goals = randint(0, goals)
        team_penalties = randint(0, penalties)
        penalty_cost = team_penalties * PENALTY_COST
        
        data['Team'].append(team)
        data['Goals'].append(team_goals)
        data['Penalty Cost'].append(penalty_cost)
    
    df = pd.DataFrame(data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.barplot(x='Team', y='Goals', data=df, ax=axes[0])
    axes[0].set_title('Goals by Team')
    
    sns.barplot(x='Team', y='Penalty Cost', data=df, ax=axes[1], palette='Blues_d')
    axes[1].set_title('Penalty Costs by Team')
    
    plt.tight_layout()
    
    return df, [axes[0], axes[1]]

# Example usage:
# df, plots = task_func(5, 3)