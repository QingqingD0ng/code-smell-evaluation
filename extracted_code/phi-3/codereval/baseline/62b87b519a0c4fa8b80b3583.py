def scale(self, other=None):
    if other is None:
        return self._scale
    if self._scale == 0:
        raise LenaValueError("Cannot rescale with zero scale")
    new_scale = other / self._scale
    self._scale = new_scale
    for field in self.fields:
        field.rescale(new_scale)
    return self