class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.freq_table = defaultdict(list)

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        value, freq = self.cache[key]
        self._update_freq_table(freq)
        return value

    def put(self, key: int, value: int) -> None:
        if self.capacity == 0:
            return
        if key in self.cache:
            self._update_freq_table(self.cache[key][1])
        else:
            if len(self.cache) >= self.capacity:
                oldest_freq, oldest_key = self._get_least_freq()
                del self.cache[oldest_key]
                self.freq_table[oldest_freq].remove(oldest_key)
            self.cache[key] = (value, 1)
            self.freq_table[1].append(key)

    def _update_freq_table(self, freq: int) -> None:
        if freq in self.freq_table:
            self.freq_table[freq].remove(key)
        self.freq_table[freq + 1].append(key)

    def _get_least_freq(self) -> (int, int):
        min_freq = min(self.freq_table.keys())
        oldest_key = self.freq_table[min_freq].pop(0)
        return min_freq, oldest_key

    def popitem(self) -> (int, int):
        if not self.cache:
            return -1, -1
        freq, key = self._get_least_freq()
        value = self.cache.pop(key)
        self._update_freq_table(freq - 1)
        return value