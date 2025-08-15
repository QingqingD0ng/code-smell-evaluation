from collections import Counter
import itertools
import string

def task_func(word: str) -> dict:
    alphabet = string.ascii_lowercase
    combinations = [''.join(pair) for pair in itertools.permutations(alphabet, 2)]
    word_counter = Counter(word)
    return {combination: word_counter.get(combination, 0) for combination in combinations}