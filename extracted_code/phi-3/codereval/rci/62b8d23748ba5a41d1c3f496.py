from collections import OrderedDict, defaultdict
import functools

def lfu_cache(maxsize=128, typed=False):
    class LFUCache:
        def __init__(self, maxsize, typed):
            self.maxsize = maxsize
            self.typed = typed
            self.cache = OrderedDict()
            self.freq = defaultdict(int)
            self.min_freq = 0

        def _evict(self):
            while self.cache and self.min_freq in self.freq and self.freq[self.min_freq] == 0:
                key, _ = self.cache.popitem(last=False)
                self.freq.pop(key)
                self.min_freq += 1

        def get(self, key):
            if key in self.cache:
                self.freq[key] += 1
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

        def put(self, key, value):
            if key in self.cache:
                self.freq[key] += 1
                self.cache.move_to_end(key)
                self.cache[key] = value
            else:
                if len(self.cache) >= self.maxsize:
                    self._evict()
                self.cache[key] = value
                self.freq[key] = 1
                self.min_freq = 1

        def __call__(self, func):
            @functools.wraps(func)
            def cached_func(*args, **kwargs):
                if self.typed:
                    key = (args, tuple(sorted(kwargs.items())))
                else:
                    key = (args, tuple(sorted(kwargs.items())))
                return self.get(key) or (self.put(key, func(*args, **kwargs)) or func(*args, **kwargs))
            return cached_func

    return LFUCache(maxsize, typed)