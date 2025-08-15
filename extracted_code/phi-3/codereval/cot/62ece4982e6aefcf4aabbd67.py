def vertex3tuple(vertices):
    return [(vertices[i], vertices[(i + 1) % len(vertices)], vertices[(i - 1) % len(vertices)]) for i in range(len(vertices))]