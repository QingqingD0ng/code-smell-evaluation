class QualityExpert:
    def __init__(self, spec_helper):
        self.spec_helper = spec_helper

    def get_deprecated_args(self):
        deprecated_args = {}
        for option in self.spec_helper.iterate_option_specs():
            if option.is_deprecated:
                deprecated_args[option.name] = option
        return deprecated_args