from zope.interface import verify, Interface, implementer

class IExample(Interface):
    def example_method(self, x, y):
        """Example method with signature"""

@implementer(IExample)
class ExampleImplementer:
    def example_method(self, x, y):
        pass

def _verify(iface, candidate, tentative=False, vtype=None):
    if not tentative and not hasattr(candidate, 'providedBy'):
        raise verify.Invalid(f"{candidate.__name__} does not claim to provide {iface.__name__}")

    missing_methods = verify.verifyMethodNames(iface, candidate) or verify.verifyAttributeNames(iface, candidate)
    if missing_methods:
        raise verify.Invalid(f"Candidate {candidate.__name__} is missing required methods or attributes: {missing_methods}")

    if vtype is not None:
        signature_errors = verify.verifyMethodSignatures(iface, candidate, vtype)
        if signature_errors:
            raise verify.Invalid(f"Candidate {candidate.__name__} methods have incorrect signatures: {signature_errors}")

    return True

# Example usage:
example_implementer = ExampleImplementer()
try:
    _verify(IExample, example_implementer)
    print("Verification passed.")
except verify.Invalid as e:
    print(f"Verification failed: {e}")