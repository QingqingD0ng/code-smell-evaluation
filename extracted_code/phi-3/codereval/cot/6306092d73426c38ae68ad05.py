class CommandOption:
    def __init__(self):
        self.option_specs = []

    def get_parser_option_specs(self):
        # Implement actual logic to retrieve parser option specs.
        # Example placeholder implementation:
        return [
            {'command':'my_command', 'option': '--my-argument', 'type': str, 'help': 'My argument description'},
            # Add more option specs as needed.
        ]

    def get_option_spec(self, command_name, argument_name):
        parser_option_specs = self.get_parser_option_specs()
        return next((spec for spec in parser_option_specs if spec['command'] == command_name and spec['option'] == argument_name), None)

# Example usage:
command_option = CommandOption()
option_spec = command_option.get_option_spec('my_command', '--my-argument')
if option_spec:
    print(option_spec)
else:
    print("Option not found.")