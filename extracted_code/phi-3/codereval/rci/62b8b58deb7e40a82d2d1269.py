from typing import List, Type

def get_provided_interfaces(obj: object) -> List[Type[Interface]]:
    provided_interfaces: List[Type[Interface]] = []
    for interface in IDeclaration.providedBy(obj):
        provided_interfaces.append(interface)
    return provided_interfaces

# Example usage:
MyInterface = Interface()
my_object = MyInterface()
interfaces = get_provided_interfaces(my_object)
print(interfaces)