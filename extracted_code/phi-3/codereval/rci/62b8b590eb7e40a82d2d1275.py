def _merge_orderings(orderings):

    if not all(hasattr(ordering, '__iter__') for ordering in orderings):

        raise ValueError("All inputs must be iterable.")

    if any(not ordering for ordering in orderings):

        raise ValueError("Input lists must not be empty.")


    merged = set()

    for ordering in orderings:

        merged.update(ordering)

    return list(merged)