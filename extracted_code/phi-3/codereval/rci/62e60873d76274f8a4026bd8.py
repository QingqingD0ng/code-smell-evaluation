from typing import Type, Optional, Dict, Tuple

class ProtocolHandler:
    """Base class for all protocol handlers."""
    pass

class HandlerV35(ProtocolHandler):
    """Handler class for version 3.5 of the protocol."""
    pass

class HandlerV40(ProtocolHandler):
    """Handler class for version 4.0 of the protocol."""
    pass

# Mapping of supported protocol version tuples to their respective handler classes
PROTOCOL_HANDLERS_MAP: Dict[Tuple[int, int], Type[ProtocolHandler]] = {
    (3, 5): HandlerV35,
    (4, 0): HandlerV40,
}

def get_supported_protocol_handlers(
    protocol_version: Optional[Tuple[int, int]] = None
) -> Dict[Tuple[int, int], Type[ProtocolHandler]]:
    """Returns a dictionary of supported protocol handlers keyed by version tuple."""
    if not isinstance(protocol_version, tuple):
        raise TypeError("protocol_version must be a tuple")

    supported_handlers: Dict[Tuple[int, int], Type[ProtocolHandler]] = {}
    for version, handler_class in PROTOCOL_HANDLERS_MAP.items():
        if protocol_version is None or version >= protocol_version:
            supported_handlers[version] = handler_class

    return supported_handlers