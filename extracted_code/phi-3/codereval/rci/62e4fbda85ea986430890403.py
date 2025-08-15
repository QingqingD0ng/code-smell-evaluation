import random

FIXED_RANDOM_SEED = 42

def _shuffled(seq: Sequence[str]) -> list[str]:
    random.seed(FIXED_RANDOM_SEED)
    return [element for element in random.sample(seq, len(seq))]