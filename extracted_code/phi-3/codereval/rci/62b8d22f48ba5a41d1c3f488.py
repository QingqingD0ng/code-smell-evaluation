def popitem(self):
    if not hasattr(self, '_data') or not isinstance(self._data, dict) or not self._data:
        raise KeyError("popitem(): dictionary is empty")
    return self._data.popitem()