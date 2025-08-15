from collections.abc import Sequence
from typing import Any, Tuple, Type

def _get_seq_with_type(seq: Any, bufsize: int = None) -> Tuple[Sequence, Type]:
    # Directly create a new sequence from the input seq
    if not isinstance(seq, Sequence):
        new_seq = type(seq)([seq])
    else:
        new_seq = seq
    seq_type = type(new_seq)
    return new_seq, seq_type