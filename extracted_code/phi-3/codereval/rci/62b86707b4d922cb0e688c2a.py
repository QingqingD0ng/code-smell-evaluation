class Registry:
    def __init__(self):
        self._hooks = {}

    def on(self, hook_name):
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []

        def decorator(func):
            def handler(*args, **kwargs):
                self._call_hooks(func, hook_name)
                return func(*args, **kwargs)
            self._hooks[hook_name].append(handler)
            return handler

        return decorator

    def _call_hooks(self, func, hook_name):
        for hook_func in self._hooks.get(hook_name, []):
            hook_func(func)

    def __getattr__(self, hook_name):
        def handler(*args, **kwargs):
            self._call_hooks(handler, hook_name)
        return handler

registry = Registry()

@registry.on('pre_execute')
def pre_execute_hook(func):
    def handler(*args, **kwargs):
        print("Pre-execute hook called before the function.")
        return func(*args, **kwargs)
    return handler

@registry.on('post_execute')
def post_execute_hook(func):
    def handler(*args, **kwargs):
        result = func(*args, **kwargs)
        print("Post-execute hook called after the function.")
        return result
    return handler

@registry.pre_execute
def my_function():
    print("My function executed.")

my_function()