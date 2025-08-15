import random

FIXED_RANDOM_SEED = 42

def _shuffled(seq: Sequence[str]) -> list[str]:
    random.seed(FIXED_RANDOM_SEED)
    shuffled_seq = list(seq)
    random.shuffle(shuffled_seq)
    return shuffled_seq