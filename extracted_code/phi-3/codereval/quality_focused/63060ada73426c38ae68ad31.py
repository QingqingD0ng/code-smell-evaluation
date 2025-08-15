def _convert_non_cli_args(self, parser_name, values_dict):
    for key, value in values_dict.items():
        if key == 'int_value':
            values_dict[key] = int(value)
        elif key == 'float_value':
            values_dict[key] = float(value)
        elif key == 'bool_value':
            values_dict[key] = value.lower() in ('true', '1', 'yes')
        elif key == 'list_value':
            values_dict[key] = value.split(',')
        elif key == 'dict_value':
            values_dict[key] = dict(item.split('=') for item in value.split(';'))
        # Add more type conversions as needed

    return values_dict