from zope.interface import verify, Invalid, Interface, Attribute

class IExampleInterface(Interface):
    """Example interface for demonstration purposes."""
    
    def example_method(self, value):
        """Method for the example interface."""
        pass
    
    example_attribute = Attribute("An example attribute for the interface.")

def verifyObject(iface, candidate, tentative=False):
    if not tentative:
        verify(iface, candidate)
    
    required_methods = iface.__dict__.get('method_required_by', [])
    required_attributes = iface.__dict__.get('attribute_required_by', [])
    
    method_errors = []
    attribute_errors = []
    
    for method_name in required_methods:
        if not hasattr(candidate, method_name):
            method_errors.append(f"Missing required method: {method_name}")
        else:
            method = getattr(candidate, method_name)
            if not callable(method):
                method_errors.append(f"Method {method_name} is not callable")
    
    for attr_name in required_attributes:
        if not hasattr(candidate, attr_name):
            attribute_errors.append(f"Missing required attribute: {attr_name}")
    
    if method_errors:
        error_message = "Method errors:\n" + "\n".join(method_errors)
        raise Invalid(error_message)
    
    if attribute_errors:
        error_message = "Attribute errors:\n" + "\n".join(attribute_errors)
        raise Invalid(error_message)
    
    return True

# Example usage:
class ExampleImplementation(Interface):
    def example_method(self, value):
        pass

    example_attribute = "Example attribute"

try:
    verifyObject(IExampleInterface, ExampleImplementation)
    print("Verification passed")
except Invalid as e:
    print(e)