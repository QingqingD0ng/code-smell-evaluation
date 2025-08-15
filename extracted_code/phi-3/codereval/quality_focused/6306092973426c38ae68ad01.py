def get_deprecated_args(self):
    deprecated_args = {}
    for option_name, option_info in self.spec_helper.iterate_option_specs().items():
        if option_info.get('deprecated', False):
            deprecated_args[option_name] = option_info
    return deprecated_args