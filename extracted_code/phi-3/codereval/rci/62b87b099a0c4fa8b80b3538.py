def is_fill_request_el(obj):
    if not isinstance(obj, object):
        return False
    return all(hasattr(obj, attr) for attr in ['fill','request'])