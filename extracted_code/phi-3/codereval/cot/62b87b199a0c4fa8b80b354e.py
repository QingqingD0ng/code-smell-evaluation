def is_fill_request_seq(seq):
    if not isinstance(seq, list) or not all(isinstance(item, int) for item in seq):
        return False
    if seq!= sorted(seq):
        return False
    return True