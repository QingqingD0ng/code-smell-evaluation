class VCSHandlerRegistry:
    def __init__(self):
        self._handlers = {}

    def register_handler(self, vcs, method):
        if vcs not in self._handlers:
            self._handlers[vcs] = {}
        self._handlers[vcs][method.__name__] = method

    def get_handler(self, vcs, method_name):
        return self._handlers.get(vcs, {}).get(method_name)

def register_vcs_handler(vcs, method):
    registry = VCSHandlerRegistry()
    registry.register_handler(vcs, method)
    return method

# Example usage:
@register_vcs_handler('git', some_git_handler_method)
def some_git_handler_method(self, *args, **kwargs):
    # implementation goes here
    pass