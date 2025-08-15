from zope.interface import Interface, implementer
from zope.component import queryUtility, adapter

class IDeclaration:
    pass

@implementer(IDeclaration)
class DirectlyProvided:
    pass

def directlyProvidedBy(obj):
    interfaces = adapter.adapt(obj, Interface)
    return interfaces

class MyInterface(Interface):
    pass

@implementer(MyInterface)
class MyClass:
    def my_method(self):
        pass

my_instance = MyClass()
provided_interfaces = directlyProvidedBy(my_instance)