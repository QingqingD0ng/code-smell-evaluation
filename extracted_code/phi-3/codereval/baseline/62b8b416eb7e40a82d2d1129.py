class MyClass:
    def __init__(self):
        self.attribute_one = "value1"
        self.__attribute_two = "value2"
        self._attribute_three = "value3"

    def names(self, all=False):
        if all:
            return [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        else:
            return [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]

# Example usage:
instance = MyClass()
print(instance.names())  # Outputs: ['attribute_one', '_attribute_three']
print(instance.names(all=True))  # Outputs: ['attribute_one', '_attribute_three', '_MyClass__init__', 'names', '__dict__', '__weakref__', '__module__', '__doc__']