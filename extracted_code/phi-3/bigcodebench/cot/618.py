import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import randint

TEAMS = ['Team A', 'Team B', 'Team C', 'Team D', 'Team E']
PENALTY_COST = 1000

def generate_match_results(goals, penalties):
    match_results = []
    for team in TEAMS:
        for opponent in TEAMS:
            if team!= opponent:
                team_goals = randint(0, goals)
                opponent_goals = randint(0, goals)
                team_penalties = randint(0, penalties)
                opponent_penalties = randint(0, penalties)
                match_result = {
                    'Team': team,
                    'Opponent': opponent,
                    'Team Goals': team_goals,
                    'Opponent Goals': opponent_goals,
                    'Team Penalties': team_penalties,
                    'Opponent Penalties': opponent_penalties,
                    'Team Penalty Costs': team_penalties * PENALTY_COST,
                    'Opponent Penalty Costs': opponent_penalties * PENALTY_COST,
                    'Result': 'Win' if team_goals > opponent_goals else 'Lose' if team_goals < opponent_goals else 'Draw'
                }
                match_results.append(match_result)
    return match_results

def visualize_results(match_results):
    df = pd.DataFrame(match_results)
    
    goals_plot = sns.barplot(x='Result', y='Team Goals', hue='Team', data=df, capsize=.05)
    plt.title('Match Goals by Result')
    plt.show()
    
    penalty_costs_plot = sns.barplot(x='Result', y='Team Penalty Costs', hue='Team', data=df)
    plt.title('Match Penalty