import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

def task_func():
    plt.rcParams.update({'font.family': 'Arial'})
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    fig = sns.pairplot(df, hue='species', vars=iris.feature_names)
    fig.fig.suptitle('Iris Dataset Pair Plot')
    fig.fig.subplots_adjust(top=0.9)
    return fig

# Example usage
# fig = task_func()
# plt.show()