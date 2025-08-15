import re


MAX_KEY_LENGTH = 20

KEY_PATTERN = r'^[a-zA-Z0-9_]{1,{MAX_KEY_LENGTH}}$'.format(MAX_KEY_LENGTH=MAX_KEY_LENGTH)


def validate_key(key):

    if not isinstance(key, str):

        raise TypeError("Key must be a string")


    return bool(re.match(KEY_PATTERN, key))