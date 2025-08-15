class InterfaceVerifier:
    def __init__(self):
        pass

    def verify_class(self, iface, candidate, tentative=False):
        """
        Verify that the *candidate* might correctly provide *iface*.
        """
        if not issubclass(candidate, iface):
            return False
        
        if not tentative and all(hasattr(candidate, attr) for attr in dir(iface)):
            return True
        
        return tentative