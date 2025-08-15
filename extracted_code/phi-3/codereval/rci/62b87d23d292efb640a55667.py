from functools import wraps

def register_vcs_handler(vcs, method):
    def decorator(func):
        func.__dict__[f'{vcs}_handler'] = method
        return wraps(func)(func)
    return decorator