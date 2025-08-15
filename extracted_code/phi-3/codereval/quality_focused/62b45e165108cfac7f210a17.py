def get_logical_path_map(inventory, version):
    path_map = {}
    for entry in inventory:
        if entry['version'] == version:
            path_map[entry['name']] = entry['path']
    return path_map