import versioneer
from typing import Any, Dict

ROOT_PATH = 'path/to/root'
AUTHOR_NAME = 'Author Name'
AUTHOR_EMAIL = 'author@example.com'

def get_config(**kwargs: Any) -> versioneer.VersioneerConfig:
    config = versioneer.VersioneerConfig()

    # Set default values using kwargs
    config.root = kwargs.get('root', ROOT_PATH)
    config.author = kwargs.get('author', AUTHOR_NAME)
    config.author_email = kwargs.get('author_email', AUTHOR_EMAIL)
    config.version = kwargs.get('version', '0.1.0')
    config.tag_version = kwargs.get('tag_version', False)
    config.write_init = kwargs.get('write_init', True)
    config.write_pyxb = kwargs.get('write_pyxb', False)
    config.write_module = kwargs.get('write_module', True)

    return config

# Example usage:
# custom_config = get_config(root='custom/root/path', author='Custom Author', version='1.0.0', tag_version=True)