def _normalizeargs(sequence, output=None):
    if output is None:
        output = []

    for item in sequence:
        if isinstance(item, tuple):
            for arg in item:
                if not isinstance(arg, str) and not hasattr(arg, 'implements') and not hasattr(arg, 'declare'):
                    output.append(arg)
                else:
                    output.append(arg)
        elif isinstance(item, str):
            output.append(item)
        else:
            output.extend(_normalizeargs(item, output))

    return output