from collections import defaultdict

def get_logical_path_map(inventory, version):
    path_map = defaultdict(list)
    for item in inventory:
        if item.get('version') == version:
            path = item.get('path', '')
            path_map[path].append(item)
    return dict(path_map)