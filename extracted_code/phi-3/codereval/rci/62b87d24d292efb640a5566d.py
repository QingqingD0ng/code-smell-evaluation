def determine_operation_sign(pieces):
    if not isinstance(pieces, list) or len(pieces) == 0:
        raise ValueError("Input must be a non-empty list.")

    last_element = pieces[-1]
    return '.' if last_element == '+' else '+'