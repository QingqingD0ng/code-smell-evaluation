class ClassInfo:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def namesAndDescriptions(self, all=False):
        if all:
            return {'name': self.name, 'description': self.description}
        else:
            return {'name': self.name, 'description': self.description}

# Example usage:
# Create an instance of ClassInfo
class_info = ClassInfo("ExampleClass", "This is an example class.")

# Get the name and description of the class attribute
info = class_info.namesAndDescriptions()
print(info)

# Get the name and description of the class attribute for all attributes (not implemented for this simple class)
info_all = class_info.namesAndDescriptions(all=True)
print(info_all)