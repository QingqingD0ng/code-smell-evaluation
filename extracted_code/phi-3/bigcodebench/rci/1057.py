import pandas as pd
import itertools
import numpy as np

def create_animal_food_pairs(animals=None, foods=None):
    default_animals = ['Dog', 'Cat', 'Elephant', 'Tiger', 'Lion', 'Zebra', 'Giraffe', 'Bear', 'Monkey', 'Kangaroo']
    default_foods = ['Meat', 'Fish', 'Grass', 'Fruits', 'Insects', 'Seeds', 'Leaves']
    
    animals = animals if animals is not None else default_animals
    foods = foods if foods is not None else default_foods
    
    combinations = list(itertools.product(animals, foods))
    np.random.shuffle(combinations)  # Efficient shuffling for large datasets
    
    df = pd.DataFrame(combinations, columns=['Animal', 'Food'], dtype=str)
    df['Pair'] = df['Animal'] + ':' + df['Food']
    df.drop(['Animal', 'Food'], axis=1, inplace=True)
    
    return df