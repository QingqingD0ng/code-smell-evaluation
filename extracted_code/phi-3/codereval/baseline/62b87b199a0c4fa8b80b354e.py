def is_fill_request_seq(seq):
    try:
        # Assuming FillRequestSeq is a class that can be instantiated with elements in seq
        FillRequestSeq(seq)
        return True
    except TypeError:
        return False