import os

class SpecParser:
    def __init__(self):
        self.include_groups = {}

    def _include_groups(self, parser_dict):
        for key, value in parser_dict.items():
            if isinstance(value, dict) and 'include' in value:
                include_path = value['include']
                if os.path.isabs(include_path):
                    with open(include_path, 'r') as include_file:
                        included_content = include_file.read()
                else:
                    included_content = self._load_spec_file(include_path)
                self.include_groups[key] = included_content

    def _load_spec_file(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()

# Usage example
spec_parser = SpecParser()
spec_parser._include_groups({
    'group1': {
        'include': '/path/to/group1_spec.yaml'
    },
    'group2': {
        'include':'relative/path/to/group2_spec.yaml'
    }
})