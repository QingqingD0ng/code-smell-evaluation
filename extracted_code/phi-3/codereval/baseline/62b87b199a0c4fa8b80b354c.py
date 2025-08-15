def _get_seq_with_type(seq, bufsize=None):
    if not hasattr(seq, '__getitem__'):
        seq = type(seq)(seq)
    seq_type = type(seq)
    return seq, seq_type