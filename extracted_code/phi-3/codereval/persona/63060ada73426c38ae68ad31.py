def _convert_non_cli_args(self, parser_name, values_dict):

    if parser_name =='main':

        for key, value in values_dict.items():

            if key in ('num1', 'num2'):

                values_dict[key] = int(value)

            elif key == 'is_active':

                values_dict[key] = value.lower() in ('true', '1')

            elif key == 'input_string':

                values_dict[key] = value

            # Add more type conversions for other keys as needed

    # Add elif blocks for other parser names with their specific type conversions