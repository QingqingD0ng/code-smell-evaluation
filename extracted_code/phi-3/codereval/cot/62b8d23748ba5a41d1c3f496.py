from collections import defaultdict
from functools import wraps, partial

class LFUCache:
    def __init__(self, maxsize=128, typed=False):
        self.maxsize = maxsize
        self.typed = typed
        self.cache = {}
        self.freq = defaultdict(int)
        self.lfu_cache = defaultdict(list)

    def _make_key(self, args, kwargs):
        if self.typed:
            key = tuple(
                (k, type(v)) for k, v in zip(args, args) +
                (kwargs.items() if kwargs else [])
            )
        else:
            key = tuple(args + (kwargs or []))
        return key

    def _evict(self):
        if not self.lfu_cache[min(self.lfu_cache, key=lambda k: self.freq[k])]:
            lfu_key = min(self.lfu_cache, key=lambda k: self.freq[k])
            del self.cache[lfu_key]
            del self.freq[lfu_key]
            del self.lfu_cache[lfu_key]

    def get(self, key):
        if key in self.cache:
            self.freq[key] += 1
            self.lfu_cache[key].append(self.cache[key])
            self._evict()
            return self.cache[key]
        return None

    def set(self, key, value):
        if key in self.cache:
            self.freq[key] += 1
            self.lfu_cache[key].append(value)
            self._evict()
        elif len(self.cache) >= self.maxsize:
            self._evict()
        self.cache[key] = value
        self.freq[key] = 1
        self.lfu_cache[key] = [value]

def lfu_cache(maxsize=128, typed=False):
    return lambda func: partial(LFUCache(maxsize, typed).