def namesAndDescriptions(self):
    current_class = self.__class__.__name__
    if all:
        return [(current_class, self.__doc__)]
    else:
        return (current_class, self.__doc__)