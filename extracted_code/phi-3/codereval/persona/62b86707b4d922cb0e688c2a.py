class HookRegistry:
    def __init__(self):
        self.hooks = {}

    def on(self, hook):
        def decorator(func):
            if hook not in self.hooks:
                self.hooks[hook] = []
            self.hooks[hook].append(func)
            return func
        return decorator

    def __call__(self, hook, *args, **kwargs):
        if hook in self.hooks:
            for handler in self.hooks[hook]:
                handler(*args, **kwargs)

registry = HookRegistry()

@registry.on('event')
def handler_for_event(*args, **kwargs):
    print("Event handler called with args:", args, "and kwargs:", kwargs)

# Usage
registry('event', 'arg1', 'arg2', key='value')  # This will trigger the handler_for_event with the provided arguments