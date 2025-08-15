class ObjectHandlerRegistry:
    def __init__(self):
        self._handlers = {}

    def register_handler(self, obj, method_name):
        def decorator(method):
            self._handlers.setdefault(obj, {})[method_name] = method
            return method
        return decorator

    def get_handler(self, obj, method_name):
        return self._handlers.get(obj, {}).get(method_name)

class VCSHandler:
    def __init__(self):
        self.registry = ObjectHandlerRegistry()

    @self.registry.register_handler(type(self),'register_vcs_handler')
    def handler_register_vcs_handler(self, vcs, method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            return method(self, vcs, *args, **kwargs)
        return wrapper

# Example usage:
# vcs_handler = VCSHandler()
# vcs_handler.register_vcs_handler('git', some_implementation)
# result = vcs_handler.register_vcs_handler('git', some_implementation)
# print(vcs_handler.get_handler(vcs_handler,'register_vcs_handler'))