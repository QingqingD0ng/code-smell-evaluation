from versioneer import VersioneerConfig

def get_config():
    config = VersioneerConfig()
    config.parent_dir = 'parent_directory_path'
    config.version = '1.0.0'
    config.script_name ='script_name.py'
    config.commit_hash = 'abc123def456'
    config.date = '2023-04-01'
    config.time = '12:34:56'
    config.vcs_url = 'https://github.com/user/repo.git'
    # Add more attributes as needed
    return config