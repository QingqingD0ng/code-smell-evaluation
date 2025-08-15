import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
import matplotlib.font_manager as fm

def task_func():
    plt.rcParams['font.family'] = 'Arial'
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    fig = sns.pairplot(df, hue="species", palette="husl")
    fig.fig.suptitle('Iris Dataset Pair Plot')
    return fig

# Example usage:
# fig = task_func()
# print(type(fig))