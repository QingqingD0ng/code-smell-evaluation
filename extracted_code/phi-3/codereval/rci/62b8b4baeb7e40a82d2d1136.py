import zope.interface
from zope.interface import Invalid
from inspect import signature

def _verify(iface, candidate, tentative=False, vtype=None):
    errors = []
    
    if not iface.provided_by(candidate) and not tentative:
        errors.append(zope.interface.Invalid(f"{candidate.__name__} does not provide {iface.__name__}"))

    for method in iface.names():
        if not hasattr(candidate, method):
            errors.append(zope.interface.Invalid(f"{candidate.__name__} does not define method {method}"))
        elif not callable(getattr(candidate, method)):
            errors.append(zope.interface.Invalid(f"{candidate.__name__}'s method {method} is not callable"))
        elif vtype and not isinstance(getattr(candidate, method), vtype):
            errors.append(zope.interface.Invalid(f"{candidate.__name__}'s method {method} does not have the correct type"))
        else:
            if not _verify_method_signature(candidate, iface, method):
                errors.append(zope.interface.Invalid(f"{candidate.__name__}'s method {method} signature does not match {iface.__name__}"))
                
    for attr in iface.names(include_methods=False):
        if not hasattr(candidate, attr):
            errors.append(zope.interface.Invalid(f"{candidate.__name__} does not define attribute {attr}"))
        elif not isinstance(getattr(candidate, attr), iface.provided_by(candidate)):
            errors.append(zope.interface.Invalid(f"{candidate.__name__}'s attribute {attr} does not have the correct type"))

    if errors:
        raise Invalid(errors)
    return True

def _verify_method_signature(candidate, iface, method_name):
    method = getattr(candidate, method_name)
    sig = signature(