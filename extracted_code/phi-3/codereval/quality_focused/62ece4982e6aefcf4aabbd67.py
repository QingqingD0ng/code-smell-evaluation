result = []
	for i in range(len(vertices)):
		left_vertex = vertices[max(i - 1, 0)]
		current_vertex = vertices[i]
		right_vertex = vertices[min(i + 1, len(vertices) - 1)]
		result.append((left_vertex, current_vertex, right_vertex))
	return result