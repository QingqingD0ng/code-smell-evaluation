def get_logical_path_map(inventory, version):
    path_map = {}
    for item in inventory:
        if item['version'] == version:
            path = item['path']
            if path in path_map:
                path_map[path].append(item)
            else:
                path_map[path] = [item]
    return path_map