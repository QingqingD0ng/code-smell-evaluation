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
    
    combinations = list(itertools.product(animals, foods))
    np.random.shuffle(combinations)
    
    df = pd.DataFrame(index=pd.Index(animals))
    for food in foods:
        df[food] = [f"{animal}:{food}" for animal, f in combinations if animal in df.index]
    
    return df