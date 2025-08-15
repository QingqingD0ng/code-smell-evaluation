def _legacy_mergeOrderings(orderings):
    merged = []
    for ordering in orderings:
        for item in ordering:
            if item not in merged:
                merged.append(item)
    return merged