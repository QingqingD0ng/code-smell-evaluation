class Registry:
    def __init__(self):
        self._hooks = {}

    def on(self, hook):
        def decorator(func):
            if hook not in self._hooks:
                self._hooks[hook] = []
            self._hooks[hook].append(func)
            return func
        return decorator

    def _call_hooks(self, func, hook):
        for h in self._hooks.get(hook, []):
            h(func)

    def __getattr__(self, hook):
        def handler(*args, **kwargs):
            self._call_hooks(handler, hook)
        return handler

registry = Registry()

# Example usage:

@registry.on('pre_execute')
def pre_execute_handler(func):
    def wrapper(*args, **kwargs):
        print("Pre-execute hook called before the function.")
        return func(*args, **kwargs)
    return wrapper

@registry.on('post_execute')
def post_execute_handler(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        print("Post-execute hook called after the function.")
        return result
    return wrapper

@registry.pre_execute
def my_function():
    print("My function executed.")

my_function()