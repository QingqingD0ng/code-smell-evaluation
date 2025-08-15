def _legacy_merge_orderings(orderings):
    merged_ordering = []
    seen = set()
    for ordering in orderings:
        for element in ordering:
            if element not in seen:
                seen.add(element)
                merged_ordering.append(element)
    return merged_ordering