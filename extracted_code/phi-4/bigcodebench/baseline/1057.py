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
    
    if not animals or not foods:
        return pd.DataFrame()
    
    combinations = list(itertools.product(animals, foods))
    np.random.shuffle(combinations)
    
    data = {food: [f"{animal}:{food}" for animal, food in combinations]}
    df = pd.DataFrame(data, index=[animal for animal, food in combinations])
    
    return df