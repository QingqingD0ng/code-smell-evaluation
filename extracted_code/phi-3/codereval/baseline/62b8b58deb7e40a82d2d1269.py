from zope.interface import Interface, implementer
from zope.interface.declarations import directlyProvides
from zope.interface.interfaces import IDeclaration

@implementer(Interface)
class MyInterface:
    pass

def directlyProvidedBy(object):
    interfaces = []
    for interface in IDeclaration.providedBy(object):
        interfaces.append(interface)
    return interfaces

# Example usage:
my_object = MyInterface()
print(directlyProvidedBy(my_object))