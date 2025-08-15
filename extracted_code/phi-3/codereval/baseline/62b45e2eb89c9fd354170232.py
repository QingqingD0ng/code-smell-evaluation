import re

def next_version(version):
    pattern = r'(\d+)\.(\d+)\.(\d+)'
    match = re.match(pattern, version)
    if match:
        major, minor, patch = map(int, match.groups())
        next_patch = patch + 1
        return f"{major}.{minor}.{next_patch}"
    else:
        raise ValueError("Invalid version format")