import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

def task_func():
    sns.set(font='Arial')
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df.columns = ['sepal length (cm)','sepal width (cm)', 'petal length (cm)', 'petal width (cm)','species']
    df['species'] = df['species'].map({0:'setosa', 1:'versicolor', 2: 'virginica'})
    g = sns.pairplot(df, hue='species')
    g.fig.suptitle('Iris Dataset Pair Plot', y=1.02)
    return g.fig