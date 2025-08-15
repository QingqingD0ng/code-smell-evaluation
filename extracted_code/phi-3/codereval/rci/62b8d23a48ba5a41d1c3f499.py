from collections import OrderedDict

class CustomOrderedDict(OrderedDict):
    def popitem(self, last=True):
        return super().popitem(last)