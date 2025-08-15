import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import numpy as np

import os


def task_func(data_url="http://lib.stat.cmu.edu/datasets/boston", save_path="heatmap.png"):

    np.random.seed(seed)

    try:

        # Load dataset

        boston = pd.read_csv(data_url)


        # Create the correlation heatmap using Seaborn

        plt.figure(figsize=(16, 10))

        sns.heatmap(boston.corr(), annot=True, fmt=".2f", cmap="coolwarm")


        # Set font family to sans-serif and Arial

        plt.rcParams['font.family'] ='sans-serif'

        plt.rcParams['font.sans-serif'] = ['Arial']


        # Option to save the heatmap to a file

        if save_path:

            plt.savefig(save_path, dpi=300)


        plt.show()


        return plt.gcf().axes[0]

    except Exception as e:

        raise ValueError(f"An error occurred: {e}")


# Use the function

ax = task_func()