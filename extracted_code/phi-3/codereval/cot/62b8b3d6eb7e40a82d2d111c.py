def _normalizeargs(sequence, output=None):
    if output is None:
        output = []

    for arg in sequence:
        if isinstance(arg, tuple) or isinstance(arg, list):
            output.extend(_normalizeargs(arg, output))
        elif isinstance(arg, str):
            output.append(arg)
        elif hasattr(arg, 'implements') or hasattr(arg, 'declarations'):
            output.extend(_normalizeargs(getattr(arg, 'implements', []), output))
            output.extend(_normalizeargs(getattr(arg, 'declarations', []), output))
        else:
            output.append(arg)

    return output