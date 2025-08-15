import pandas as pd
import itertools
import numpy as np

def task_func(animals=None, foods=None):
    default_animals = ['Dog', 'Cat', 'Elephant', 'Tiger', 'Lion', 'Zebra', 'Giraffe', 'Bear', 'Monkey', 'Kangaroo']
    default_foods = ['Meat', 'Fish', 'Grass', 'Fruits', 'Insects', 'Seeds', 'Leaves']
    
    animals = animals or default_animals
    foods = foods or default_foods
    
    if not animals or not foods:
        return pd.DataFrame()
    
    combinations = itertools.product(animals, foods)
    pairs = [f"{animal}:{food}" for animal, food in combinations]
    np.random.shuffle(pairs)
    
    df = pd.DataFrame(pairs, columns=foods)
    df.index.name = 'Animal'
    
    return df