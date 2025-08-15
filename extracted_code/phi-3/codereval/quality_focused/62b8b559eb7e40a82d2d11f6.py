class DetermineMetaclass:
    @staticmethod
    def determine_metaclass(bases, explicit_mc=None):
        # Check for explicit metaclass
        if explicit_mc is not None:
            return explicit_mc

        # Check if all bases inherit from a common metaclass
        metaclasses = set()
        for base in bases:
            if hasattr(base, '__metaclass__'):
                metaclasses.add(base.__metaclass__)
            elif hasattr(base, '__class__'):
                metaclasses.add(base.__class__)

        if len(metaclasses) == 1:
            return metaclasses.pop()

        # Default to type if no metaclass is found
        return type

# Example usage:
# Assuming we have a base class with a metaclass
class BaseWithMetaclass(type):
    pass

# Class with explicit metaclass
class ExplicitMetaclass(type):
    pass

class MyClass(metaclass=DetermineMetaclass):
    pass

# Class inheriting from two classes with different metaclasses
class MyBase(BaseWithMetaclass):
    pass

class MyDerived(MyBase, ExplicitMetaclass):
    pass

# Infer metaclass for MyClass
print(DetermineMetaclass.determine_metaclass(MyClass.__bases__))  # Should print BaseWithMetaclass

# Infer metaclass for MyDerived
print(DetermineMetaclass.determine_metaclass(MyDerived.__bases__))  # Should print ExplicitMetaclass