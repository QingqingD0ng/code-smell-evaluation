import os
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpecParser:
    def __init__(self):
        self.include_groups = {}

    def _load_yaml_file(self, file_path):
        """Loads YAML content from a file."""
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f'File not found: {file_path}')
            return None
        except (yaml.YAMLError, IOError) as e:
            logger.error(f'Error reading file {file_path}: {e}')
            return None

    def _validate_include_path(self, file_path):
        """Validates YAML include path."""
        if not os.path.isabs(file_path):
            if not os.path.exists(os.path.join(os.path.dirname(__file__), file_path)):
                return False
        return True

    def _include_groups(self, parser_dict):
        """Resolves include directives in spec files."""
        for key, value in parser_dict.items():
            if isinstance(value, dict) and 'include' in value:
                include_path = value['include']
                if self._validate_include_path(include_path):
                    included_content = self._load_yaml_file(include_path)
                    if included_content is not None:
                        self.include_groups[key] = included_content
                else:
                    logger.warning(f'Invalid include path: {include_path}')

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