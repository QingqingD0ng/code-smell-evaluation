from collections import Counter
import itertools
import string

def task_func(word: str) -> dict:
    lowercase_alphabets = string.ascii_lowercase
    combinations = [''.join(p) for p in itertools.product(lowercase_alphabets, repeat=2)]
    word_combinations = [word[i:i+2] for i in range(len(word) - 1)]
    word_combination_counts = Counter(word_combinations)
    
    return {comb: word_combination_counts.get(comb, 0) for comb in combinations}