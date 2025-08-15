class MyClass:
    pass

def subclasses(cls):
    for subclass in cls.__subclasses__():
        yield subclass
        for subclass in subclasses(subclass):
            yield subclass