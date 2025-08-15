import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

TEAMS = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E']
PENALTY_COST = 1000

def task_func(goals, penalties):
    results = []
    for team in TEAMS:
        for _ in range(5):  # Assuming 5 matches per team
            goals_scored = random.randint(0, goals)
            penalties_received = random.randint(0, penalties)
            fine = penalties_received * PENALTY_COST
            results.append({'Team': team, 'Goals Scored': goals_scored, 'Goals Conceded': 10 - goals_scored, 'Penalties Received': penalties_received, 'Fine': fine})

    df = pd.DataFrame(results)

    goals_plot = sns.barplot(x='Team', y='Goals Scored', data=df, palette='viridis')
    goals_plot.set_title('Goals Scored by Team')
    goals_plot.set(xlabel='Team', ylabel='Goals Scored')

    penalties_plot = sns.barplot(x='Team', y='Fine', data=df, palette='magma')
    penalties_plot.set_title('Fines Received by Team')
    penalties_plot.set(xlabel='Team', ylabel='Fine')

    return df, [goals_plot, penalties_plot]

df, plots = task_func(5, 3)
plt.show()