class SomeObject:
    def fill(self):
        # Implementation of fill method
        pass

    def compute(self):
        # Implementation of compute method
        pass


def is_fill_compute_el(obj):
    method_exists = all(hasattr(obj, method) for method in ('fill', 'compute'))
    return method_exists and callable(getattr(obj, 'fill', None)) and \
           callable(getattr(obj, 'compute', None))


# Example usage:
some_obj = SomeObject()
print(is_fill_compute_el(some_obj))  # Output: True