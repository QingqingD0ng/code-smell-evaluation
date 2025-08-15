def get_spec_defaults(self):
    defaults = {}
    for arg_name, arg_spec in self._arg_specs.items():
        defaults[arg_name] = self._get_defaults(arg_spec)
    return defaults