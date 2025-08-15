import re


def next_version(version):

    major, minor, patch = map(int, re.split('\.|\-', version))

    return f"{major}.{minor}.{patch + 1}"