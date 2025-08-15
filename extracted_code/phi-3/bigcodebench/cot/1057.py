import pandas as pd
import itertools
import numpy as np

def task_func(animals=None, foods=None):
    predefined_animals = ['Dog', 'Cat', 'Elephant', 'Tiger', 'Lion', 'Zebra', 'Giraffe', 'Bear', 'Monkey', 'Kangaroo']
    predefined_foods = ['Meat', 'Fish', 'Grass', 'Fruits', 'Insects', 'Seeds', 'Leaves']
    
    if not animals and not foods:
        return pd.DataFrame()
    
    animals = animals if animals else predefined_animals
    foods = foods if foods else predefined_foods
    
    combinations = list(itertools.product(animals, foods))
    np.random.shuffle(combinations)
    
    df = pd.DataFrame(combinations, columns=['Animal', 'Food']).apply(lambda x: f"{x[0]}:{x[1]}", axis=1)
    
    return df