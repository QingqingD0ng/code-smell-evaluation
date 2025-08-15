import collections

def _get_seq_with_type(seq, bufsize=None):
    if bufsize is not None:
        seq = seq[:bufsize]
    seq_type = type(seq)
    if seq_type.__module__ == 'collections' and issubclass(seq_type, collections.abc.Sequence):
        return seq, seq_type
    return seq, type(seq)