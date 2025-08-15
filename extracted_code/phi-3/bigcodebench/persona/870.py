import pandas as pd
import numpy as np

def task_func(data_list=[('a', 1, 2.1), ('b', 2, 3.2), ('c', 3, 4.3), ('d', 4, 5.4), ('e', 5, 6.5)]):
    means = [np.nan] * len(next(iter(data_list)))
    for tuple_ in data_list:
        for i, value in enumerate(tuple_):
            try:
                means[i] = means[i] + float(value)
            except ValueError:
                continue
    means = [mean / len(data_list) if not np.isnan(mean) else np.nan for mean in means]
    df = pd.DataFrame({'Mean Value': means}, index=[f'Position {i}' for i in range(len(means))])
    return df