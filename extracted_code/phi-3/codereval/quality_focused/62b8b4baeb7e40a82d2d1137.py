from zope.interface import Interface, implementedBy, verify

class MyInterface(Interface):
    def method_one():
        pass

    def method_two(arg1):
        pass

def verify_object(iface, candidate, tentative=False):
    if not implementedBy(candidate, iface):
        raise verify.Invalid(f"Candidate class {candidate.__name__} does not implement the interface {iface.__name__}")

    missing_methods = [method for method in iface.__dict__ if not hasattr(candidate, method)]
    if missing_methods:
        raise verify.Invalid(f"Candidate class {candidate.__name__} is missing methods: {', '.join(missing_methods)}")

    incorrect_signatures = []
    for method in iface.__dict__:
        if hasattr(candidate, method):
            candidate_method = getattr(candidate, method)
            if not callable(candidate_method):
                incorrect_signatures.append((method, candidate_method))

    if incorrect_signatures:
        raise verify.Invalid(f"Candidate class {candidate.__name__} has incorrect signatures: {', '.join([f'{method}: {sig}' for method, sig in incorrect_signatures])}")

    return True