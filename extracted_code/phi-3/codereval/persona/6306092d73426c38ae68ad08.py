def _get_conditionally_required_args(self, command_name, options_spec, args):

    required_when_args = []

    for option in options_spec:

        if option.get('required_when', False) and option.get('name') in args:

            required_when_args.extend(option.get('required_when', []))

    return required_when_args