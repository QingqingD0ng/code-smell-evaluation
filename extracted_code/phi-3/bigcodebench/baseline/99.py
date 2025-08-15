import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties

import pandas as pd

import seaborn as sns

from sklearn.datasets import load_iris


def task_func():
    plt.rcParams['font.family'] = 'Arial'
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    g = sns.pairplot(df, hue='species')
    g.fig.suptitle('Iris Dataset Pair Plot')
    return plt.gcf()


fig = task_func()