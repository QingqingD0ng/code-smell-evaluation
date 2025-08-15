import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import numpy as np

import os


def task_func(data_url="http://lib.stat.cmu.edu/datasets/boston", seed=42):

    np.random.seed(seed)

    try:

        # Load the Boston Housing dataset

        boston = pd.read_csv(data_url, delimiter=r"\s+")


        # Create a correlation matrix

        corr = boston.corr()


        # Set up the matplotlib font

        plt.rcParams['font.family'] = ['sans-serif', 'Arial']


        # Plot the heatmap

        sns.set(style="white")

        f, ax = plt.subplots(figsize=(11, 9))

        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5, ax=ax)

        plt.title('Boston Housing Dataset Correlation Heatmap')


        # Save the heatmap to a file

        plt.savefig('boston_housing_correlation.png', bbox_inches='tight')


        return ax


    except Exception as e:

        raise ValueError(f"An error occurred: {e}")