class MyClass:
    def __init__(self):
        self._attribute_one = "value1"
        self.__attribute_two = "value2"
        self._attribute_three = "value3"

    @property
    def attribute_one(self):
        return self._attribute_one

    @property
    def attribute_two(self):
        return self.__attribute_two

    @property
    def attribute_three(self):
        return self._attribute_three

    def attribute_names(self, all=False):
        """
        Return the attribute names of current class.
        """
        if all:
            return [attr for attr in vars(self) if not callable(getattr(self, attr, None)) and not attr.startswith("__")]
        else:
            return [attr for attr in vars(self) if not callable(getattr(self, attr, None)) and not attr.startswith("__")]

# Example usage:
instance = MyClass()
print(instance.attribute_names())  # Outputs: ['attribute_one', 'attribute_three']
print(instance.attribute_names(all=True))  # Outputs: ['attribute_one', 'attribute_two', 'attribute_three']