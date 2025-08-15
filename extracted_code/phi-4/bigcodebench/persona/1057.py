import pandas as pd
import itertools
import numpy as np

def task_func(animals=None, foods=None):
    default_animals = ['Dog', 'Cat', 'Elephant', 'Tiger', 'Lion', 'Zebra', 'Giraffe', 'Bear', 'Monkey', 'Kangaroo']
    default_foods = ['Meat', 'Fish', 'Grass', 'Fruits', 'Insects', 'Seeds', 'Leaves']
    
    if animals is None:
        animals = default_animals
    if foods is None:
        foods = default_foods
    
    if not animals and not foods:
        return pd.DataFrame()
    
    if not animals:
        animals = default_animals
    if not foods:
        foods = default_foods
    
    combinations = list(itertools.product(animals, foods))
    np.random.shuffle(combinations)
    
    df = pd.DataFrame(combinations, columns=['Animal', 'Food'])
    df['Animal_Food'] = df['Animal'] + ':' + df['Food']
    
    return df.pivot(columns='Food', values='Animal_Food').reset_index(drop=True)