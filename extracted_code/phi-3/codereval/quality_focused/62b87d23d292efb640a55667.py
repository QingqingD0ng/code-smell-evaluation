def register_vcs_handler(vcs, method):
    def decorator(func):
        func.vcs_handler = vcs
        func.method = method
        return func
    return decorator