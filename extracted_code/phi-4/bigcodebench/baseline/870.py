import pandas as pd
import numpy as np
import itertools

def task_func(data_list=[('a', 1, 2.1), ('b', 2, 3.2), ('c', 3, 4.3), ('d', 4, 5.4), ('e', 5, 6.5)]):
    if not data_list:
        return pd.DataFrame()

    transposed = list(zip(*data_list))
    means = []

    for i, items in enumerate(transposed):
        numeric_items = [x for x in items if isinstance(x, (int, float))]
        if numeric_items:
            mean_value = np.mean(numeric_items)
        else:
            mean_value = np.nan
        means.append(mean_value)

    df = pd.DataFrame(means, index=[f'Position {i}' for i in range(len(means))], columns=['Mean Value'])
    return df