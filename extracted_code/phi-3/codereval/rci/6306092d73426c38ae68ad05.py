from typing import Any, Dict, Optional

class CommandParser:
    def get_parser_option_specs(self) -> Dict[str, Dict[str, Any]]:
        # Implementation of retrieving parser's option specs
        pass

    def get_option_spec(self, command_name: str, argument_name: str) -> Optional[Dict[str, Any]]:
        parser_spec = self.get_parser_option_specs()
        return parser_spec.get(command_name, {}).get(argument_name)

# Example usage:
# parser = CommandParser()
# option_spec = parser.get_option_spec('my_command','my_option')
# if option_spec is not None:
#     print("Option specification found:", option_spec)
# else:
#     print("Option specification not found.")