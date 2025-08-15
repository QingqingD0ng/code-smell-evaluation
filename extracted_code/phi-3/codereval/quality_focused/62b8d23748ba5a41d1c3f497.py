def popitem(self):
    if self.is_empty():
        raise KeyError("popitem(): dictionary is empty")
    min_key = min(self.frequency, key=self.frequency.get)
    value = self.data[min_key]
    del self.data[min_key]
    del self.frequency[min_key]
    return min_key, value