class LFUCache:
    LEAST_FREQUENT = float('inf')

    def __init__(self):
        self.cache = {}
        self.freq_map = {}
        self.min_freq = LFUCache.LEAST_FREQUENT
        self.min_key = None

    def get(self, key):
        if key in self.cache:
            self._update_freq(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        self.cache[key] = value
        self._update_freq(key)

    def _update_freq(self, key):
        self.freq_map[key] = self.freq_map.get(key, 0) + 1
        if self.freq_map[key] < self.min_freq:
            self.min_freq = self.freq_map[key]
            self.min_key = key

    def popitem(self):
        if self.min_key is not None:
            self.freq_map.pop(self.min_key)
            self.cache.pop(self.min_key)
            self.min_freq = LFUCache.LEAST_FREQUENT
            self.min_key = None
        else:
            raise ValueError("Cache is empty")

    def _move_to_last(self, key):
        value = self.cache.pop(key)
        self.cache[key] = value
        self._update_freq(key)

    def increase_freq(self, key):
        self._update_freq(key)
        self._move_to_last(key)