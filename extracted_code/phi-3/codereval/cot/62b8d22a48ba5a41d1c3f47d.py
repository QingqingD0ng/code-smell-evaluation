class CustomDict:
    def __init__(self):
        self.__data = {}
    
    def __getitem__(self, key):
        return self.__data[key]
    
    def __setitem__(self, key, value):
        self.__data[key] = value
    
    def __delitem__(self, key):
        del self.__data[key]
    
    def __contains__(self, key):
        return key in self.__data
    
    def pop(self, key, default=object()):
        return self.__data.pop(key, default)

# Example usage:
# d = CustomDict()
# d['foo'] = 'bar'
# value = d.pop('foo')  # value would be 'bar'
# value = d.pop('nonexistent', 'default')  # value would be 'default'