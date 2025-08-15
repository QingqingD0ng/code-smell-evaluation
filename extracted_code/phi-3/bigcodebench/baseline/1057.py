import pandas as pd
import itertools
import numpy as np

def task_func(animals=None, foods=None):
    animals = animals if animals is not None else ['Dog', 'Cat', 'Elephant', 'Tiger', 'Lion', 'Zebra', 'Giraffe', 'Bear', 'Monkey', 'Kangaroo']
    foods = foods if foods is not None else ['Meat', 'Fish', 'Grass', 'Fruits', 'Insects', 'Seeds', 'Leaves']
    
    # Generate all combinations
    combinations = list(itertools.product(animals, foods))
    np.random.shuffle(combinations)  # Shuffle combinations
    
    # Create DataFrame
    df = pd.DataFrame(combinations, columns=['Animal', 'Food'])
    df['Pair'] = df['Animal'] + ':' + df['Food']
    df.drop(['Animal', 'Food'], axis=1, inplace=True)
    
    return df