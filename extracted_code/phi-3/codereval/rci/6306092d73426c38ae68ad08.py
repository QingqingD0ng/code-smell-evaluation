def _get_conditionally_required_args(self, command_name, options_spec, args):
    conditionally_required_args = []
    for option in options_spec:
        if option.get('name') == command_name and'required_when' in option:
            conditionally_required_args.extend(
                [arg for arg in option['required_when'] if arg in args]
            )
    return conditionally_required_args