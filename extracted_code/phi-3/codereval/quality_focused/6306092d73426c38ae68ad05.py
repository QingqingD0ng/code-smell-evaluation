class OptionSpecFetcher:
    def __init__(self, parser_specifications):
        self.parser_specifications = parser_specifications

    def get_option_spec(self, command_name, argument_name):
        return self.get_parser_option_specs()[command_name].get(argument_name)

    def get_parser_option_specs(self):
        return self.parser_specifications