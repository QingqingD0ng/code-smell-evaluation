from collections import Counter
import itertools
import string

def task_func(word: str) -> dict:
    alphabet = string.ascii_lowercase
    two_letter_combinations = [''.join(pair) for pair in itertools.combinations(alphabet, 2)]
    word_counter = Counter(word)
    return {comb: word_counter.get(comb, 0) for comb in two_letter_combinations}