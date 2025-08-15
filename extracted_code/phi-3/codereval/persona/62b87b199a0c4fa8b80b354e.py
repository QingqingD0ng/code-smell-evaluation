def is_fill_request_seq(seq):
    return all(isinstance(item, (int, float)) for item in seq) and len(seq) > 0