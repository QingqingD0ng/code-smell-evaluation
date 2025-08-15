import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

def task_func():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    plt.rcParams.update({'font.family': 'Arial'})

    figure = sns.pairplot(df, hue='species', 
                          vars=[iris.feature_names[0], iris.feature_names[1], 
                                iris.feature_names[2], iris.feature_names[3]])

    figure.fig.suptitle('Iris Dataset Pair Plot', y=1.02)

    return figure.fig