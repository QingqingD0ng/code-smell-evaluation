def is_fill_compute_el(obj):
    return callable(getattr(obj, 'fill', None)) and callable(getattr(obj, 'compute', None))