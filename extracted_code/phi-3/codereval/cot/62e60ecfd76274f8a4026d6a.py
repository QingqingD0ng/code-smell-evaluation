class ProtocolHandlerFactory:
    @classmethod
    def create_handler(cls, protocol_version=None):
        if protocol_version is None:
            return cls.default_handler
        elif protocol_version == 1:
            return cls.handler_for_version_1
        elif protocol_version == 2:
            return cls.handler_for_version_2
        else:
            raise ValueError(f"Unsupported protocol version: {protocol_version}")
    
    @classmethod
    def default_handler(cls):
        # Default handler implementation
        pass
    
    @classmethod
    def handler_for_version_1(cls):
        # Version 1 handler implementation
        pass
    
    @classmethod
    def handler_for_version_2(cls):
        # Version 2 handler implementation
        pass