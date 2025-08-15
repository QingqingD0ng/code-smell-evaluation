import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

def task_func():
    # Load the Iris dataset
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target

    # Set global font to Arial
    plt.rcParams['font.family'] = 'Arial'

    # Create a pair plot
    pair_plot = sns.pairplot(df, hue='species', diag_kind='kde', palette='Dark2')

    # Set titles and labels
    pair_plot.fig.suptitle('Iris Dataset Pair Plot', fontsize=16, y=1.02)
    for ax, col in zip(pair_plot.diag_axes, iris.feature_names):
        ax.set_title(col)

    return pair_plot.fig