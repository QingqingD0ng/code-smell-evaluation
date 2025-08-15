from packaging.version import parse

def next_version(version_str):
    current_version = parse(version_str)
    next_version = Version(str(current_version.major) + '.' + str(current_version.minor) + '.' + str(current_version.patch + 1))
    return next_version.base_version