from zope.interface import Interface, implementer, directlyProvidedBy

@implementer(Interface)
class MyInterface:
    pass

@implementer(Interface)
class MyImplementation:
    pass

def get_direct_interfaces(obj):
    return directlyProvidedBy(obj)

my_interface = MyInterface()
my_implementation = MyImplementation()

print(get_direct_interfaces(my_interface))
print(get_direct_interfaces(my_implementation))