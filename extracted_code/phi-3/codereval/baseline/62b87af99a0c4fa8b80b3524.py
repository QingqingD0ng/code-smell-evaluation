def is_run_el(obj):
    return hasattr(obj, 'run') and callable(getattr(obj, 'run'))