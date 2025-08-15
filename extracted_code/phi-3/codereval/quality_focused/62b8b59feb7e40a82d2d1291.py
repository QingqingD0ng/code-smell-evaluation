class YourClass:
    def _get_target_class(self):
        # Assuming the implementation in use is stored in an attribute named 'implementation'
        implementation = getattr(self, 'implementation', 'default_implementation')
        return implementation.replace('Py', '').replace('Fallback', '')

# Example usage:
your_instance = YourClass()
target_class = your_instance._get_target_class()
print(target_class)