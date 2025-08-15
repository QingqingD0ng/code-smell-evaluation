def create_vertex_neighbor_tuples(vertices):
    vertex_neighbor_tuples = []
    for index, vertex in enumerate(vertices):
        prev_index = (index - 1) % len(vertices)
        next_index = (index + 1) % len(vertices)
        prev_vertex = vertices[prev_index]
        current_vertex = vertices[index]
        next_vertex = vertices[next_index]
        vertex_neighbor_tuples.append((prev_vertex, current_vertex, next_vertex))
    return vertex_neighbor_tuples