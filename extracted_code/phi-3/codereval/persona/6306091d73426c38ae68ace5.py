import os

def _include_groups(self, parser_dict):
    for key, value in parser_dict.items():
        if isinstance(value, dict) and 'include' in value:
            include_path = value['include']
            if os.path.isfile(include_path):
                with open(include_path, 'r') as include_file:
                    included_parser = include_file.read()
                    parser_dict[key].update(eval(included_parser))
            elif os.path.isdir(include_path):
                for filename in os.listdir(include_path):
                    if filename.endswith('.py'):
                        file_path = os.path.join(include_path, filename)
                        with open(file_path, 'r') as file:
                            file_contents = file.read()
                            included_parser = file_contents
                            parser_dict[key].update(eval(included_parser))
            else:
                raise FileNotFoundError(f"Include path {include_path} does not exist.")