from zope.interface import Interface, implementer, providedBy, Invalid

class IExampleInterface(Interface):
    def example_method(self, param1, param2):
        pass

@implementer(IExampleInterface)
class ExampleImplementation:
    def example_method(self, param1, param2):
        pass

def verify_object(iface, candidate, tentative=False):
    if not providedBy(candidate, iface) and not tentative:
        raise Invalid(f"Candidate does not claim to provide {iface}")
    
    required_methods = [method for method in dir(iface) if callable(getattr(iface, method, None)) and not method.startswith("__")]
    required_attributes = [attr for attr in dir(iface) if not callable(getattr(iface, attr, None)) and not attr.startswith("__")]
    
    method_errors = [
        f"{candidate.__class__.__name__}.{method}" for method in required_methods
        if not hasattr(candidate, method) or not callable(getattr(candidate, method))
    ]
    attribute_errors = [
        f"{candidate.__class__.__name__}.{attr}" for attr in required_attributes
        if not hasattr(candidate, attr)
    ]
    
    errors = method_errors + attribute_errors
    if errors:
        error_messages = "; ".join(errors)
        raise Invalid(f"Candidate object is missing required methods/attributes: {error_messages}")
    
    return True

# Example usage:
iface = IExampleInterface()
candidate = ExampleImplementation()

try:
    verify_object(iface, candidate)
    print("Candidate object passes verification.")
except Invalid as e:
    print(f"Verification failed: {e}")