from collections import Counter
import itertools
import string

def task_func(word: str) -> dict:
    alphabet = string.ascii_lowercase
    two_letter_combinations = [''.join(p) for p in itertools.product(alphabet, repeat=2)]
    word_lower = word.lower()
    word_counter = Counter(word_lower[i:i+2] for i in range(len(word_lower) - 1))
    
    result = {comb: word_counter[comb] for comb in two_letter_combinations}
    return result