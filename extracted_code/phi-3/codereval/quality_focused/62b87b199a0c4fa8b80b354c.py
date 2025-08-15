def _get_seq_with_type(seq, bufsize=None):
    seq_type = type(seq)
    if bufsize is not None and hasattr(seq, '__getitem__'):
        return seq[:bufsize], seq_type
    return seq, seq_type