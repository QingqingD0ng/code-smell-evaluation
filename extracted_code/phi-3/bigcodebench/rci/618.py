import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

NUM_MATCHES = 5
PENALTY_COST = 1000

def generate_match_results(num_teams, max_goals, max_penalties, num_matches):
    team_names = [f'Team {i+1}' for i in range(num_teams)]
    results = []

    for team in team_names:
        for _ in range(num_matches):
            goals_scored = random.randint(0, max_goals)
            penalties_received = random.randint(0, max_penalties)
            fine = penalties_received * PENALTY_COST
            results.append({
                'Team': team,
                'Goals Scored': goals_scored,
                'Goals Conceded': 10 - goals_scored,
                'Penalties Received': penalties_received,
                'Fine': fine
            })

    df = pd.DataFrame(results)
    return df

def visualize_match_results(df):
    goals_plot = sns.barplot(x='Team', y='Goals Scored', data=df, palette='viridis')
    goals_plot.set_title('Goals Scored by Team')
    goals_plot.set(xlabel='Team', ylabel='Goals Scored')

    penalties_plot = sns.barplot(x='Team', y='Fine', data=df, palette='magma')
    penalties_plot.set_title('Fines Received by Team')
    penalties_plot.set(xlabel='Team', ylabel='Fine')

    plt.show()

def main():
    df = generate_match_results(5, 5, 3, NUM_MATCHES)
    visualize_match_results(df)

if __name__ == "__main__":
    main()