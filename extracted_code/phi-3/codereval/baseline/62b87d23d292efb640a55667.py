def register_vcs_handler(vcs, method):
    def decorator(func):
        func.__dict__[vcs] = method
        return func
    return decorator