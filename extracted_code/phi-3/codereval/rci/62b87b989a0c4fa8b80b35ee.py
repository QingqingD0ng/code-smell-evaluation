class YourClass:
    def __init__(self, initial_value=None):
        # Constructor initializes the context and bins for the instance
        self.context = {}
        self.bins = initial_value if initial_value is not None else self.make_bins()

    def make_bins(self):
        # Creates initial bins for an instance. This is an instance method.
        # Example implementation (to be replaced with actual logic):
        return {'bin1': 1, 'bin2': 2}

    @classmethod
    def reset_class(cls):
        # Class method to reset the state for all instances of the class
        for instance in cls._instances:  # Assuming _instances is a class variable holding all instances
            instance.reset_instance()
        cls.bins = cls.make_bins()  # Rebuilds the initial state for all instances

    def reset_instance(self):
        # Instance method to reset the context and bins of this particular instance
        self.context = {}
        self.bins = self.make_bins()

# Example usage:
# Create an instance of YourClass with initial bins value
instance_a = YourClass(initial_value={'bin1': 100, 'bin2': 200})
# Reset the state of the class (all instances)
YourClass.reset_class()
# Reset the state of an individual instance
instance_a.reset_instance()