def _normalizeargs(sequence, output=None):
    if output is None:
        output = []

    def normalize_item(item):
        if isinstance(item, Declaration):
            return [item]
        elif isinstance(item, (list, tuple)):
            return [normalize_item(subitem) for subitem in item]
        elif isinstance(item, str):
            return [item]
        else:
            raise ValueError("Unsupported type in sequence")

    for item in sequence:
        normalized_item = normalize_item(item)
        output.extend(normalized_item)

    return output