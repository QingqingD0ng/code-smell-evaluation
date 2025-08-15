class CommandParser:
    def get_parser_option_specs(self, command_name):
        parser_options = {
           'main': [
                {'name': '--help', 'action': 'help', 'help': 'Show this help message and exit'},
                # Add more options for'main' command here
            ],
            'virsh': [
                {'name': '--version', 'action':'version','version': 'VIRSH version 2.2.11', 'help': 'Show version information and exit'},
                {'name': '--list', 'action':'store_true', 'help': 'List all available domains'},
                # Add more options for 'virsh' command here
            ],
            'ospd': [
                {'name': '--list', 'action':'store_true', 'help': 'List all OpenPBS nodes'},
                # Add more options for 'ospd' command here
            ],
            # Add more command-specific options dictionaries here
        }

        return parser_options.get(command_name, [])