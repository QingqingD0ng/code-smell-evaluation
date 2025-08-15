def get_deprecated_args(self):
    return {option: details for option, details in self.spec_helper.iterate_option_specs() if details['deprecated']}