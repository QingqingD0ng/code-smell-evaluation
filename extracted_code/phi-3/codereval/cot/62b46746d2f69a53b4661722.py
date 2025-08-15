def absorb(args):
    result = []
    for expr in args:
        absorbed = False
        for i in range(len(expr) - 1):
            if expr[i] == expr[i + 1]:
                result.append(expr[i] * 2)
                absorbed = True
                break
        if not absorbed:
            result.append(expr)
    return result