import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

TEAMS = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E']
PENALTY_COST = 1000  # in dollars

def task_func(goals, penalties):
    goals_scored = [random.randint(0, goals) for _ in range(len(TEAMS) * 2)]
    goals_received = goals_scored[:]
    random.shuffle(goals_received)
    
    penalty_costs = [random.randint(0, penalties) for _ in range(len(TEAMS) * 2)]
    
    matches = pd.DataFrame({
        'Home Team': [TEAMS[i // 2] for i in range(len(TEAMS) * 2)],
        'Away Team': [TEAMS[(i + 1) // 2] for i in range(len(TEAMS) * 2)],
        'Goals Scored': goals_scored,
        'Goals Received': goals_received,
        'Penalty Cost': [PENALTY_COST * pc for pc in penalty_costs]
    })
    
    goals_plot = sns.lineplot(data=matches, x='Match Number', y='Goals Scored', hue='Home Team')
    penalty_costs_plot = sns.barplot(data=matches, x='Match Number', y='Penalty Cost')
    
    plt.show()
    return matches, [goals_plot, penalty_costs_plot]