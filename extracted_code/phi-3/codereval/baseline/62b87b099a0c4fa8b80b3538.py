def is_fill_request_el(obj):
    return hasattr(obj, 'fill') and hasattr(obj,'request')