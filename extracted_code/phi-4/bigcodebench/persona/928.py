from collections import Counter
import itertools
import string

def task_func(word: str) -> dict:
    alphabets = string.ascii_lowercase
    all_combinations = [''.join(pair) for pair in itertools.permutations(alphabets, 2)]
    word_combinations = [''.join(word[i:i+2]) for i in range(len(word) - 1)]
    combination_counts = Counter(word_combinations)
    return {combo: combination_counts.get(combo, 0) for combo in all_combinations}