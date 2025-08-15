def get_logical_path_map(inventory, version):
    path_map = {}
    for entry in inventory:
        file_path = entry.get('path', '')
        if file_path.startswith('/'):
            file_path = '/' + file_path.lstrip('/')
        if version in entry and'states' in entry[version]:
            for state in entry[version]['states']:
                path_map[state] = file_path
    return path_map