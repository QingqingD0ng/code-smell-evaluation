from collections import OrderedDict

def lfu_cache(maxsize=128, typed=False):
    class LFUCache:
        def __init__(self, maxsize, typed):
            self.maxsize = maxsize
            self.typed = typed
            self.cache = OrderedDict()
            self.freq = {}
            self.min_freq = 0

        def get(self, key):
            if key in self.cache:
                self.freq[key] += 1
                self.freq[key] = min(self.freq[key], self.maxsize)
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

        def put(self, key, value):
            if key in self.cache:
                self.freq[key] += 1
                self.freq[key] = min(self.freq[key], self.maxsize)
                self.cache.move_to_end(key)
                self.cache[key] = value
            else:
                if len(self.cache) >= self.maxsize:
                    self.cache.popitem(last=False)
                self.cache[key] = value
                if not self.freq:
                    self.min_freq = 1
                else:
                    self.min_freq = min(self.freq.values())

        def __call__(self, func):
            def wrapped(*args, **kwargs):
                if self.typed:
                    key = (args, tuple(sorted(kwargs.items())))
                else:
                    key = (args, tuple(sorted(kwargs.items())))
                if key not in self.cache:
                    result = func(*args, **kwargs)
                    self.put(key, result)
                return self.get(key)
            return wrapped

    return LFUCache(maxsize, typed)