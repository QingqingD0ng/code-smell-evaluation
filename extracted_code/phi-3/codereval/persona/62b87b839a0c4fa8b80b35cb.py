def _get_err_indices(self, coord_name):
    indices = []
    for i, coord in enumerate(self.coordinates):
        if coord[0] == coord_name:
            indices.append(i)
    return indices