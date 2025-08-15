def get_option_spec(self, command_name, argument_name):
    return self.get_parser_option_specs().get(command_name, {}).get(argument_name)