class InterfaceVerifier:
    def __init__(self):
        self.required_methods = []

    def _extract_required_methods(self, iface):
        for method in iface.__dict__.values():
            if callable(method) and not method.__name__.startswith('_'):
                self.required_methods.append(method.__name__)

    def _check_candidate_methods(self, candidate):
        for method_name in self.required_methods:
            if method_name not in candidate.__dict__ or not callable(candidate.__dict__[method_name]):
                return False
        return True

    def verifyClass(self, iface, candidate, tentative=False):
        if not isinstance(candidate, type):  # Verify candidate is a class
            raise TypeError("Candidate must be a class.")

        self._extract_required_methods(iface)
        if self._check_candidate_methods(candidate) or tentative:
            return True

        missing_methods = [method for method in self.required_methods if method not in candidate.__dict__]
        return False, missing_methods