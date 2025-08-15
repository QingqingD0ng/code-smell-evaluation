class MyDict(dict):
    def popitem(self):
        return dict.popitem(self)