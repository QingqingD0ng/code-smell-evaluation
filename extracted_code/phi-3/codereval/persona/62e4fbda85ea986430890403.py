import random

FIXED_RANDOM_SEED = 42

def _shuffled(seq: Sequence[str]) -> list[str]:
    random.seed(FIXED_RANDOM_SEED)
    shuffled_seq = random.sample(seq, len(seq))
    return shuffled_seq