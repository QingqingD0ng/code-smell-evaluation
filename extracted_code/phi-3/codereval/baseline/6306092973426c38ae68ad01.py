class YourClass:  # Replace with your actual class name
    def get_deprecated_args(self):
        deprecated_args = {}
        for option in self.spec_helper.iterate_option_specs():
            if option.is_deprecated:
                deprecated_args[option.name] = option.message
        return deprecated_args