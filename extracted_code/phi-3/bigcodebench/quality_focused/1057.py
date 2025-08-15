import pandas as pd
import itertools
import numpy as np

def task_func(animals=None, foods=None):
    if not animals and not foods:
        return pd.DataFrame()
    animals = animals or ['Dog', 'Cat', 'Elephant', 'Tiger', 'Lion', 'Zebra', 'Giraffe', 'Bear', 'Monkey', 'Kangaroo']
    foods = foods or ['Meat', 'Fish', 'Grass', 'Fruits', 'Insects', 'Seeds', 'Leaves']
    combinations = list(itertools.product(animals, foods))
    df = pd.DataFrame(combinations, columns=['Animal', 'Food'])
    df['Pair'] = df['Animal'] + ':' + df['Food']
    np.random.shuffle(df.to_numpy())
    df.reset_index(drop=True, inplace=True)
    return df