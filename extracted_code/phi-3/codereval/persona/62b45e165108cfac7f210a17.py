from typing import Dict, List

def get_logical_path_map(inventory: Dict[str, List[str]], version: str) -> Dict[str, str]:
    path_map = {}
    for state in inventory:
        for file in inventory[state]:
            if file.endswith(f'.{version}'):
                path_map[file] = f"{state}/{file}"
    return path_map