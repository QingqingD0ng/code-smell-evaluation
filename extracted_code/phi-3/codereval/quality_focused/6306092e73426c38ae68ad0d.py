def create_complex_argument_type(self, subcommand, type_name, option_name, spec_option):

    return COMPLEX_TYPES.get(type_name, lambda *args, **kwargs: None)(

        *[getattr(self, attr, None) for attr in ['vars', 'defaults', 'plugin_path']],

        subcommand, spec_option

    )