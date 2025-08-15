def subclasses(cls):
    return cls.__subclasses__() + [c for s in cls.__subclasses__() for c in subclasses(s)]