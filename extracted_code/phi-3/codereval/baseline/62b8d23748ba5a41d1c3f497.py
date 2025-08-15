class LFUPopitem:
    def __init__(self):
        self.cache = {}
        self.freq_map = {}
        self.min_freq = float('inf')
        self.min_key = None

    def get(self, key):
        if key in self.cache:
            self._update_freq(key)
            return self.cache[key]
        else:
            return None

    def put(self, key, value):
        self.cache[key] = value
        self._update_freq(key)

    def _update_freq(self, key):
        if key in self.freq_map:
            self.freq_map[key] += 1
        else:
            self.freq_map[key] = 1

        if self.freq_map[key] < self.min_freq:
            self.min_freq = self.freq_map[key]
            self.min_key = key

    def popitem(self):
        if self.min_key is not None:
            self.freq_map.pop(self.min_key)
            self.cache.pop(self.min_key)
            self.min_freq = float('inf')
            self.min_key = None
            return self.min_key, self.cache[self.min_key]
        else:
            raise KeyError("Cache is empty")