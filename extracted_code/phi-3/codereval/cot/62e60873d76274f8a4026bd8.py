class BoltProtocolHandlers:
    handlers = {
        (3, 5): HandlerClass35,
        (3, 6): HandlerClass36,
        # Add other versions and their corresponding handler classes here
    }

    @classmethod
    def protocol_handlers(cls, protocol_version=None):
        if protocol_version is not None and not isinstance(protocol_version, tuple):
            raise TypeError("protocol_version must be a tuple")
        
        if protocol_version is None:
            return {version: cls.handlers[version] for version in cls.handlers}
        elif protocol_version in cls.handlers:
            return {protocol_version: cls.handlers[protocol_version]}
        else:
            return {}