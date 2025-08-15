import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from matplotlib.font_manager import FontProperties

def task_func():
    # Set the global font to Arial
    plt.rcParams['font.family'] = 'Arial'
    
    # Load the iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target
    
    # Create the pair plot
    fig, ax = plt.subplots()
    sns.pairplot(iris_df, hue='species', palette='viridis', diag_kind='hist', markers='+')
    
    # Set the title and labels
    ax.set_title('Iris Dataset Pair Plot')
    ax.set_xlabels(['sepal length (cm)','sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    ax.set_ylabels(['sepal length (cm)','sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    
    return fig