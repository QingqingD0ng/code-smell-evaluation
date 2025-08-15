class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.access_counter = Counter()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.access_counter[key] += 1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        self.access_counter[key] += 1
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.access_counter[oldest_key]

    def popitem(self):
        if not self.cache:
            raise KeyError("popitem(): dictionary is empty")
        oldest_key, _ = self.cache.popitem(last=False)
        return oldest_key