from collections import Counter
import itertools
import string

def task_func(word: str) -> dict:
    alphabet = string.ascii_lowercase
    combos = Counter(itertools.permutations(alphabet, 2))
    counts = {combo: word.count(combo) for combo in combos}
    return counts