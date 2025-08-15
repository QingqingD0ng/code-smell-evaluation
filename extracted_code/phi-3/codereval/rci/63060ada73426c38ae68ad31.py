import ast

def convert_arguments(arg_dict):
    for key, value in list(arg_dict.items()):
        if key!= 'args':
            try:
                arg_dict[key] = int(value)
            except ValueError:
                try:
                    arg_dict[key] = float(value)
                except ValueError:
                    try:
                        arg_dict[key] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        raise ValueError(f"Cannot convert argument '{key}' to a recognized type")
    return arg_dict