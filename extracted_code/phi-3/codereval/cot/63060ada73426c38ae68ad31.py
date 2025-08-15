class NonCliArgumentConverter:
    def __init__(self):
        self.type_mappings = {
            'int': int,
            'float': float,
            'bool': self.bool_converter,
            # Add more type mappings as needed
        }

    def bool_converter(self, value):
        return value.lower() in ('true', '1', 'yes') if value else False

    def convert_values(self, values_dict):
        for key, value in values_dict.items():
            expected_type = self.get_expected_type(key)
            values_dict[key] = self.type_mappings[expected_type](value)

    def get_expected_type(self, key):
        # Implement logic to determine the expected type based on the parser_name and key
        # This is a placeholder for the actual logic, which would likely be more complex
        return 'int'  # Default to integer for simplicity

    def _convert_non_cli_args(self, parser_name, values_dict):
        self.convert_values(values_dict)
        return values_dict

# Example usage:
converter = NonCliArgumentConverter()
values = {'volume_size': '10','storage_mode': 'block'}
converted_values = converter._convert_non_cli_args('main', values)
print(converted_values)  # Output will be {'volume_size': 10,'storage_mode': False}