import re
from typing import Tuple

def parse_version(version: str) -> Tuple[int, int, int]:
    pattern = r'(\d+)\.(\d+)\.(\d+)'
    match = re.match(pattern, version)
    if match:
        return tuple(map(int, match.groups()))
    else:
        raise ValueError("Invalid version format")

def increment_patch_version(major: int, minor: int, patch: int) -> Tuple[int, int, int]:
    return major, minor, patch + 1

def next_version(version: str) -> str:
    major, minor, patch = parse_version(version)
    new_major, new_minor, new_patch = increment_patch_version(major, minor, patch)
    return f"{new_major}.{new_minor}.{new_patch}"

# Example usage:
try:
    print(next_version("1.2.3"))  # Output: 1.2.4
    print(next_version("2.0.0"))  # Output: 2.0.1