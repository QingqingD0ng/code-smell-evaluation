class MyClass:
    attribute = "This is an example attribute"

    @classmethod
    def class_attribute_descriptions(cls, all=False):
        all_attributes = cls.__dict__.copy()
        filtered_attributes = [(name, value) for name, value in all_attributes.items() if not name.startswith('__')]
        return filtered_attributes if all else filtered_attributes

# Example usage:
print(MyClass.class_attribute_descriptions())
print(MyClass.class_attribute_descriptions(all=True))