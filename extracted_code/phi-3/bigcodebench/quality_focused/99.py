import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

# Set the global font to Arial
plt.rcParams['font.family'] = 'Arial'

def task_func():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target

    # Create a seaborn pair plot
    fig = sns.pairplot(df, hue='species', palette='bright')

    # Set the title of the plot
    fig.fig.suptitle('Iris Dataset Pair Plot')

    # Set the x-axis and y-axis labels
    for ax in fig.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=45)
        plt.setp(ax.get_yticklabels(), rotation=45)

    return fig

fig = task_func()