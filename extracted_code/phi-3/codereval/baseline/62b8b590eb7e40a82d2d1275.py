# Assuming the input is a list of lists
def _legacy_merge_orderings(orderings):
    merged = []
    for ordering in orderings:
        merged.extend(ordering)
    return list(dict.fromkeys(merged))