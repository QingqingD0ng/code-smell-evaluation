def convert_key_to_int(self, key):
    if not isinstance(key, (int, str)):
        raise TypeError(f"Key must be an integer or convertible to one, got {type(key)}")

    try:
        return int(key)
    except ValueError:
        raise TypeError(f"Key '{key}' cannot be converted to an integer") from None