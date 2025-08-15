import pandas as pd
import itertools
import numpy as np

def task_func(animals=None, foods=None):
    if animals is None:
        animals = ['Dog', 'Cat', 'Elephant', 'Tiger', 'Lion', 'Zebra', 'Giraffe', 'Bear', 'Monkey', 'Kangaroo']
    if foods is None:
        foods = ['Meat', 'Fish', 'Grass', 'Fruits', 'Insects', 'Seeds', 'Leaves']
    combinations = list(itertools.product(animals, foods))
    np.random.shuffle(combinations)
    df = pd.DataFrame(combinations, columns=['Animal', 'Food'])
    df['Pair'] = df['Animal'] + ':' + df['Food']
    return df

# Example usage:
animal_food_pairs = task_func(['Dog', 'Cat'], ['Meat', 'Fish'])
print(animal_food_pairs)