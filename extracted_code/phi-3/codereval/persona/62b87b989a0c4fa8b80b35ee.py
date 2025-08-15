class QualityExpert:
    def __init__(self, initial_value=None, make_bins=None):
        self.context = {}
        self.bins = make_bins() if make_bins else initial_value

    def reset(self):
        self.context = {}
        self.bins = self.bins() if callable(self.bins) else self.bins