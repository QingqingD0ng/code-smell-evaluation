def has_run_method(target_obj):
    return hasattr(target_obj, 'run') and callable(getattr(target_obj, 'run', None))

def is_run_el(target_obj):
    return has_run_method(target_obj)