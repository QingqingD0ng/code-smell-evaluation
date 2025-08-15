from versioneer import VersioneerConfig

def get_config():
    config = VersioneerConfig()
    config.root = 'path/to/root'
    config.author = 'Author Name'
    config.author_email = 'author@example.com'
    config.version = '0.1.0'
    config.tag_version = False
    config.write_init = True
    config.write_pyxb = False
    config.write_module = True
    return config