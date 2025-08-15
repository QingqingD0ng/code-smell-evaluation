from collections import Counter
import itertools
import string

def task_func(word: str) -> dict:
    word = word.lower()
    all_combinations = [''.join(p) for p in itertools.permutations(string.ascii_lowercase, 2)]
    word_combinations = Counter(''.join(word[i:i+2] for i in range(len(word)-1)))
    return {comb: word_combinations.get(comb, 0) for comb in all_combinations}