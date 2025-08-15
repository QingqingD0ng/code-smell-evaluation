vertices_3tuple = []

for i in range(len(vertices)):
    prev_index = (i - 1) % len(vertices)
    next_index = (i + 1) % len(vertices)
    prev_vertex = vertices[prev_index]
    current_vertex = vertices[i]
    next_vertex = vertices[next_index]
    vertices_3tuple.append((prev_vertex, current_vertex, next_vertex))

return vertices_3tuple