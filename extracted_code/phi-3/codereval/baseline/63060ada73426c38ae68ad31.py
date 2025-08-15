import ast

def _convert_non_cli_args(self, parser_name, values_dict):
    for key, value in values_dict.items():
        if key!= 'args':  # Assuming 'args' is a special key for CLI args
            try:
                # Attempt to convert to int
                values_dict[key] = int(value)
            except ValueError:
                try:
                    # Attempt to convert to float
                    values_dict[key] = float(value)
                except ValueError:
                    # Attempt to parse as literal (list, dict, etc.)
                    values_dict[key] = ast.literal_eval(value)