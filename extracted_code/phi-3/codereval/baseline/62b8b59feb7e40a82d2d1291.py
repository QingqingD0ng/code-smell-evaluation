def _getTargetClass(self):
    # Assuming there's an attribute self.target_class that holds the class name
    return self.target_class.rstrip("Py").rstrip("Fallback")