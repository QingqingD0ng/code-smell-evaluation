from collections.abc import Sequence

def _get_seq_with_type(seq, bufsize=None):
    if isinstance(seq, Sequence) or seq is None:
        return seq, type(seq)
    return None, None