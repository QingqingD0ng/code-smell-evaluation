class ProtocolHandlersRegistry:
    handlers = {}

    @classmethod
    def register_handler(cls, version, handler_cls):
        cls.handlers[version] = handler_cls

    @classmethod
    def get_handlers(cls, protocol_version=None):
        if protocol_version is not None:
            if not isinstance(protocol_version, tuple):
                raise TypeError("Protocol version must be a tuple")
            return {protocol_version: cls.handlers.get(protocol_version)}
        else:
            return cls.handlers

# Example usage:
ProtocolHandlersRegistry.register_handler((3, 5), MyBoltProtocolV35Handler)
ProtocolHandlersRegistry.register_handler((4, 0), MyBoltProtocolV4Handler)

# To retrieve all handlers, call without arguments:
all_handlers = ProtocolHandlersRegistry.get_handlers()

# To retrieve a specific handler, pass the version tuple:
specific_handler = ProtocolHandlersRegistry.get_handlers((3, 5))