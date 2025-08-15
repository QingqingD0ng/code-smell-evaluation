class RunEligibilityChecker:
    @staticmethod
    def has_run_method(obj):
        return hasattr(obj, 'run') and callable(getattr(obj, 'run'))

class SomeClassWithRun:
    def run(self):
        print("Running the method.")

checker = RunEligibilityChecker()
print(checker.has_run_method(SomeClassWithRun()))  # Output: True

class SomeClassWithoutRun:
    def some_other_method(self):
        pass

print(checker.has_run_method(SomeClassWithoutRun()))  # Output: False