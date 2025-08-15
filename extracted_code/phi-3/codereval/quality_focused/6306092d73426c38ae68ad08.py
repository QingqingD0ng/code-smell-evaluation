def _get_conditionally_required_args(self, command_name, options_spec, args):
    conditionally_required_args = []
    for option in options_spec:
        if option.get('required_when', False) and (not option.get('name') or option['name'] not in args):
            conditionally_required_args.append(option['name'])
    return conditionally_required_args