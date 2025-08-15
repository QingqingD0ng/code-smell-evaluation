from zope.interface import Interface, implementer, implementedBy

# Assuming the object has some method to determine its interfaces
def directly_provided_by(obj):
    interfaces = get_direct_interfaces(obj)
    declarations = []
    for interface in interfaces:
        for implementation in implemented_by(interface):
            declarations.append(implementation)
    return declarations

# Dummy function to represent getting direct interfaces
def get_direct_interfaces(obj):
    # Replace with actual logic to determine direct interfaces
    return []

# Dummy function to represent checking implemented interfaces
def implemented_by(interface):
    # Replace with actual logic to check implemented interfaces
    return []