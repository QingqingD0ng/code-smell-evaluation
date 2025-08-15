def is_none_string(val):
    """Check if a string represents a None value."""
    if isinstance(val, str):
        return val.strip().lower() == 'none'
    return False