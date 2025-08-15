class MyClass:
    def run(self):
        pass

class AnotherClass:
    def do_something(self):
        pass

def is_run_method(obj):
    return hasattr(obj, 'run') and callable(getattr(obj, 'run'))

# Example usage
my_obj = MyClass()
another_obj = AnotherClass()

print(is_run_method(my_obj))  # Output: True
print(is_run_method(another_obj))  # Output: False