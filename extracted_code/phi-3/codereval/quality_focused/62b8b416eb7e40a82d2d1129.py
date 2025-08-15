class MyClass:
    def names(self, all=False):
        if all:
            return self.__dict__
        else:
            return [key for key in self.__dict__ if not key.startswith('_')]