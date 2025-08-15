import numpy as np

import pandas as pd

import matplotlib.pyplot as plt


COLUMNS = ['A', 'B', 'C', 'D', 'E']


def task_func(a, b):

    df = pd.DataFrame(np.random.rand(len(a), len(b)), index=a, columns=b)

    ax = df.plot(kind='bar')

    return ax