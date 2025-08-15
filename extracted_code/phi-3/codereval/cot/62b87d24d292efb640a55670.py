def get_versions(default_version='1.0.0'):
    try:
        with open('version.json', 'r') as file:
            data = json.load(file)
            return data['version']
    except (FileNotFoundError, KeyError):
        return default_version