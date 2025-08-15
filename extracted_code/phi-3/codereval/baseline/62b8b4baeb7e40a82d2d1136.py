import zope.interface
from zope.interface import Invalid

def _verify(iface, candidate, tentative=False, vtype=None):
    errors = []
    if not iface.provided_by(candidate):
        errors.append(zope.interface.Invalid(f"{candidate.__name__} does not provide {iface.__name__}"))
    elif not tentative:
        methods = iface.names()
        for method in methods:
            if not hasattr(candidate, method):
                errors.append(zope.interface.Invalid(f"{candidate.__name__} does not define method {method}"))
            elif not callable(getattr(candidate, method)):
                errors.append(zope.interface.Invalid(f"{candidate.__name__}'s method {method} is not callable"))
            elif vtype and not isinstance(getattr(candidate, method), vtype):
                errors.append(zope.interface.Invalid(f"{candidate.__name__}'s method {method} does not have the correct type"))
        attributes = iface.names(include_methods=False)
        for attr in attributes:
            if not hasattr(candidate, attr):
                errors.append(zope.interface.Invalid(f"{candidate.__name__} does not define attribute {attr}"))
            elif not isinstance(getattr(candidate, attr), iface.provided_by(candidate)):
                errors.append(zope.interface.Invalid(f"{candidate.__name__}'s attribute {attr} does not have the correct type"))

    if errors:
        raise Invalid(errors)
    return True