class MyClass:
    attribute1 = "value1"
    attribute2 = "value2"
    attribute3 = "value3"

    @staticmethod
    def names(cls, all=False):
        return {
            attr
            for attr in dir(cls)
            if not callable(getattr(cls, attr)) and not attr.startswith("__")
        }

# Example usage:
my_class_instance = MyClass()
print(MyClass.names())  # Output: {'attribute1', 'attribute2', 'attribute3'}