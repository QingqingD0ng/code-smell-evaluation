class SomeObject:
    def fill(self):
        pass

    def compute(self):
        pass


def is_fill_compute_el(obj):
    return hasattr(obj, 'fill') and callable(getattr(obj, 'fill')) and \
           hasattr(obj, 'compute') and callable(getattr(obj, 'compute'))


# Example usage:
some_obj = SomeObject()
print(is_fill_compute_el(some_obj))  # Output: True