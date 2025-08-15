def _normalizeargs(sequence, output=None, seen=None):
    if output is None:
        output = []
    if seen is None:
        seen = set()

    def normalize(item):
        if id(item) in seen:
            return
        seen.add(id(item))

        if isinstance(item, (tuple, list)):
            for arg in item:
                normalize(arg)
                if not isinstance(arg, str) and not hasattr(arg, 'implements') and not hasattr(arg, 'declare'):
                    output.append(arg)
        elif isinstance(item, str):
            output.append(item)
        else:
            seen.add(id(item))
            normalize(item)

    for item in sequence:
        normalize(item)

    return output