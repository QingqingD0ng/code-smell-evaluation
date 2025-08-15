def _get_conditionally_required_args(self, command_name, options_spec, args):

    return [

        arg['name'] for arg in options_spec

        if command_name == arg.get('name') and'required_when' in arg and any(

            cond(args) for cond in arg['required_when']

        )

    ]