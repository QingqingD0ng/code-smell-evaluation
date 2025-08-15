class YourClass:
    def __init__(self, initial_value=None):
        self.context = {}
        self.bins = initial_value if initial_value is not None else self.make_bins()

    def make_bins(self):
        # Implementation of make_bins method
        pass

    def reset(self):
        self.context = {}
        self.bins = self.bins.__class__(self.bins) if hasattr(self.bins, '__class__') else self.make_bins()