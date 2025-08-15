class QualityExpert:
    def names(self, all=False):
        return [attr for attr in dir(self) if not attr.startswith('_') and all]