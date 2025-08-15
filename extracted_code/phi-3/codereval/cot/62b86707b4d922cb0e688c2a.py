class HookRegistry:
    def __init__(self):
        self._hooks = {}

    def register_hook(self, hook, handler):
        if hook not in self._hooks:
            self._hooks[hook] = []
        self._hooks[hook].append(handler)

    def get_hooks(self, hook):
        return self._hooks.get(hook, [])

    def on(self, hook):
        def decorator(handler):
            self.register_hook(hook, handler)
            return handler
        return decorator