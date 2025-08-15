def vertex3tuple(vertices):
    result = []
    for i, vertex in enumerate(vertices):
        prev = vertices[i - 1] if i > 0 else vertices[-1]
        next_ = vertices[(i + 1) % len(vertices)]
        result.append((vertex, prev, next_))
    return result