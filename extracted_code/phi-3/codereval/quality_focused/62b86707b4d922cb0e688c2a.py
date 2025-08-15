class HookRegistry:
    def __init__(self):
        self._hooks = {}

    def on(self, hook_name):
        def decorator(func):
            if hook_name not in self._hooks:
                self._hooks[hook_name] = []
            self._hooks[hook_name].append(func)
            return func
        return decorator

    def run_hooks(self, hook_name, *args, **kwargs):
        if hook_name in self._hooks:
            for func in self._hooks[hook_name]:
                func(*args, **kwargs)