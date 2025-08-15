from collections import Counter
import itertools
import string

def task_func(word: str) -> dict:
    alphabet = string.ascii_lowercase
    two_letter_combinations = [''.join(pair) for pair in itertools.permutations(alphabet, 2)]
    return dict(Counter(two_letter_combinations) & Counter(word))