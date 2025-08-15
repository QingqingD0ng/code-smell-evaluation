class MyClass:
    attribute = "This is an example attribute"

    @staticmethod
    def namesAndDescriptions(all=False):
        if all:
            return [(attr, MyClass.__dict__[attr]) for attr in MyClass.__dict__ if not attr.startswith('__')]
        else:
            return [(attr, MyClass.__dict__[attr]) for attr in MyClass.__dict__ if not attr.startswith('__') or attr == 'namesAndDescriptions']

# Example usage:
my_class_instance = MyClass()
print(my_class_instance.namesAndDescriptions())
print(my_class_instance.namesAndDescriptions(all=True))