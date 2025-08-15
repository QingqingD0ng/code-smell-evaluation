from enum import Enum

class ProtocolVersion(Enum):
    V0 = 0
    V1 = 1
    V2 = 2

class UnsupportedProtocolVersionError(Exception):
    """Exception raised for unsupported protocol versions."""
    pass

class BoltProtocolHandlerFactory:
    @classmethod
    def create_handler(cls, protocol_version):
        if protocol_version not in ProtocolVersion:
            raise UnsupportedProtocolVersionError(f"Unsupported protocol version: {protocol_version}")
        return {
            ProtocolVersion.V0: BoltProtocolHandlerV0.create_handler(),
            ProtocolVersion.V1: BoltProtocolHandlerV1.create_handler(),
            ProtocolVersion.V2: BoltProtocolHandlerV2.create_handler(),
        }[protocol_version]

class BoltProtocolHandlerV0:
    @classmethod
    def create_handler(cls):
        # Implement the handler for version 0 of the Bolt protocol
        pass

class BoltProtocolHandlerV1:
    @classmethod
    def create_handler(cls):
        # Implement the handler for version 1 of the Bolt protocol
        pass

class BoltProtocolHandlerV2:
    @classmethod
    def create_handler(cls):
        # Implement the handler for version 2 of the Bolt protocol
        pass