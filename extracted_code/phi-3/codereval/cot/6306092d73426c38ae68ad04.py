class CommandOptionParser:
    def __init__(self):
        self.options = {
          'main': ['--verbose', '--version'],
            'virsh': ['--help', '--list', '--detail'],
            'ospd': ['--reset', '--status'],
            # Add other commands and their options here
        }

    def get_parser_option_specs(self, command_name):
        return self.options.get(command_name.lower(), [])

# Example usage:
parser = CommandOptionParser()
options_for_main = parser.get_parser_option_specs('main')
print(options_for_main)