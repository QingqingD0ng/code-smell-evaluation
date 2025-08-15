def _normalize_args(sequence, output=None):
    normalized = []
    for item in sequence:
        if isinstance(item, str):
            # Assuming Declarations, tuples, or single interfaces are strings
            normalized.append(item)
        elif isinstance(item, tuple):
            # Expand the tuple into individual items
            normalized.extend(item)
        elif isinstance(item, dict):
            # Expand dictionary into a list of items
            normalized.extend(item.values())
        else:
            # Assuming the item is already a Declaration or interface
            normalized.append(item)
    return normalized if output is None else output.extend(normalized)