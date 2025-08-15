class BoltProtocolHandlerFactory:
    @classmethod
    def protocol_handlers(cls, protocol_version=None):
        if protocol_version is None or protocol_version == 0:
            return cls._create_handler_v0()
        elif protocol_version == 1:
            return cls._create_handler_v1()
        elif protocol_version == 2:
            return cls._create_handler_v2()
        else:
            raise ValueError("Unsupported protocol version")

    @staticmethod
    def _create_handler_v0():
        # Implement the handler for version 0 of the Bolt protocol
        pass

    @staticmethod
    def _create_handler_v1():
        # Implement the handler for version 1 of the Bolt protocol
        pass

    @staticmethod
    def _create_handler_v2():
        # Implement the handler for version 2 of the Bolt protocol
        pass