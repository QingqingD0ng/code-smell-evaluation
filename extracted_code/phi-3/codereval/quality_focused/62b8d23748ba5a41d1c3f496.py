from collections import defaultdict, OrderedDict
from functools import wraps

class LFUCache:
    def __init__(self, maxsize=128, typed=False):
        self.maxsize = maxsize
        self.typed = typed
        self.cache = OrderedDict()
        self.access_counter = defaultdict(int)
        self.freq_counter = defaultdict(int)

    def _get(self, key):
        if self.typed:
            key = self._type_key(key)
        if key not in self.cache:
            return None
        self.access_counter[key] += 1
        self.freq_counter[len(self.access_counter)] += 1
        self.cache.move_to_end(key)
        return self.cache[key]

    def _type_key(self, key):
        if self.typed:
            return (key, type(key))
        return key

    def _put(self, key, value):
        if self.typed:
            key = self._type_key(key)
        if key in self.cache:
            self.cache.move_to_end(key)
            self.access_counter[key] += 1
        elif len(self.cache) >= self.maxsize:
            min_freq = min(self.freq_counter)
            keys_to_evict = [k for k, v in self.access_counter.items() if v == min_freq]
            for k in keys_to_evict:
                self.cache.pop(k)
                self.freq_counter[len(self.access_counter)] -= 1
        self.cache[key] = value
        self.access_counter[key] += 1
        self.freq_counter[len(self.access_counter)] += 1

    def __call__(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            key = self._type_key(args + tuple(sorted(kwargs.items())))
            if key in self.cache: