import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


def parse_data(data):
    values = pd.Series(data.split('-')).astype(float)
    if values.empty:
        raise ValueError("Input data must contain at least one value.")
    return values


def calculate_bins(values):
    return np.arange(values.min(), values.max() + 2) - 0.5


def plot_histogram(values, bins):
    ax = values.hist(bins=bins)
    ax.set_title('Histogram of Values')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    return ax


def task_func(data):
    values = parse_data(data)
    bins = calculate_bins(values)
    ax = plot_histogram(values, bins)
    return ax