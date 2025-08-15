from collections import Counter
import itertools
import string

def task_func(word: str) -> dict:
    alphabets = string.ascii_lowercase
    combos = [''.join(combo) for combo in itertools.product(alphabets, repeat=2)]
    word_combos = Counter(itertools.combinations(word, 2))
    return {combo: word_combos.get(combo, 0) for combo in combos}