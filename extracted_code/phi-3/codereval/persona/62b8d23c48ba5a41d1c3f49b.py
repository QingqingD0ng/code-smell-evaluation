class LRUCache:
    def __init__(self, capacity):
        self.cache = {}
        self.capacity = capacity
        self.count = 0
        self.order = []

    def get(self, key):
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        else:
            return -1

    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif self.count == self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)
        self.count += 1

    def popitem(self):
        if self.order:
            oldest = self.order.pop(0)
            return oldest, self.cache.pop(oldest)
        else:
            return None