def _legacy_mergeOrderings(orderings):
    merged = []
    for ordering in orderings:
        for element in ordering:
            if element not in merged:
                merged.append(element)
    return merged