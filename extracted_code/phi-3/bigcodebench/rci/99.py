import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties

import pandas as pd

import seaborn as sns

from sklearn.datasets import load_iris


def load_dataset(dataset_loader):
    return dataset_loader()


def set_global_font(font_name):
    plt.rcParams['font.family'] = font_name

def create_pair_plot(df, title):
    g = sns.pairplot(df, hue='species')
    g.fig.suptitle(title)
    return g.fig


def task_func(font_name='Arial'):
    set_global_font(font_name)
    iris = load_dataset(load_iris)
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    fig = create_pair_plot(df, 'Iris Dataset Pair Plot')
    return fig


fig = task_func()