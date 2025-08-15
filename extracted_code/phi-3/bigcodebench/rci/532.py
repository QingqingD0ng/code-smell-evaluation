import numpy as np

from collections import Counter

from scipy.stats import norm

import matplotlib.pyplot as plt


def calculate_duplicates(df):

    return Counter(df['value'].values)


def plot_histogram(data, bins, title, x_label, y_label):

    plt.hist(data, bins=bins, color='green', alpha=0.6)

    plt.title(title)

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    plt.show()


def fit_and_plot_normal_distribution(data, ax):

    mu, std = norm.fit(data)

    xmin, xmax = ax.get_xlim()

    x = np.linspace(xmin, xmax, 100)

    p = norm.pdf(x, mu, std)

    ax.plot(x, p, 'k', linewidth=2)


def task_func(df, bins=4):

    if df.empty:

        return Counter(), plt.figure().add_subplot(111)

    duplicates = calculate_duplicates(df)

    non_duplicates = {val: count for val, count in duplicates.items() if count > 1}

    if not non_duplicates:

        print("No non-duplicate values found.")

        return Counter(), plt.figure().add_subplot(111)

    plot_histogram(df['value'], bins, 'Distribution', 'Value', 'Frequency')

    ax = plt.gca()

    fit_and_plot_normal_distribution(df['value'], ax)

    return non_duplicates, ax