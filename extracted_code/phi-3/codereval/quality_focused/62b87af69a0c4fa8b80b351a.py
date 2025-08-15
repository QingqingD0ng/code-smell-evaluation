class MyClass:
    # Example class with fill and compute methods
    def fill(self):
        pass

    def compute(self):
        pass

def is_fill_compute_el(obj):
    return all(hasattr(obj, method) for method in ['fill', 'compute'])

# Example usage:
# my_instance = MyClass()
# print(is_fill_compute_el(my_instance))  # Should return True if MyClass has both methods

# Another class without one of the methods
class MyOtherClass:
    def fill(self):
        pass

    def another_method(self):
        pass

# print(is_fill_compute_el(MyOtherClass()))  # Should return False