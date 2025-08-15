import inspect
from zope.interface import Interface, implementer, providedBy, Attribute
from zope.interface.exceptions import Invalid

def verify_object(iface, candidate, tentative=False):
    verify_provided_by(iface, candidate, tentative)
    verify_methods(candidate, iface)
    verify_attributes(candidate, iface)
    return True

def verify_provided_by(iface, candidate, tentative):
    if not providedBy(candidate, iface) and not tentative:
        raise Invalid(f"{candidate} must provide {iface}")

def verify_methods(candidate, iface):
    candidate_methods = {name: method for name, method in inspect.getmembers(candidate, inspect.isfunction)}
    required_methods = iface.methods()

    missing_methods = required_methods - set(candidate_methods.keys())
    if missing_methods:
        raise Invalid(f"Candidate is missing methods: {missing_methods}")

    for method_name, method in candidate_methods.items():
        if_method = getattr(iface, method_name)

        if not callable(method):
            raise Invalid(f"{method_name} is not callable or not a method in {candidate}")

        if not if_method.isabstract:
            try:
                signature = inspect.signature(method)
                if if_method.signature!= signature:
                    raise Invalid(f"Signature for {method_name} does not match: {signature}!= {if_method.signature}")
            except ValueError:
                raise Invalid(f"Signature for {method_name} could not be determined")

def verify_attributes(candidate, iface):
    required_attributes = {attr.name for attr in iface.attributes()}
    candidate_attributes = {attr for attr in dir(candidate) if not attr.startswith('__')}

    missing_attributes = required_attributes - candidate_attributes
    if missing_attributes:
        raise Invalid(f"Candidate is missing attributes: