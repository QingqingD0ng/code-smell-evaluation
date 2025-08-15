import warnings
from typing import Tuple, Dict, Type, Optional

# Assuming handler classes are defined somewhere like this:
class HandlerV35(ProtocolHandler): pass
class HandlerV40(ProtocolHandler): pass

protocol_handlers_map = {
    (3, 5): HandlerV35,
    (4, 0): HandlerV40,
}

class ProtocolHandler:
    pass

def protocol_handlers(cls: Type[ProtocolHandler], protocol_version: Optional[Tuple[int, int]] = None) -> Dict[Tuple[int, int], Type[ProtocolHandler]]:
    if protocol_version is not None and not isinstance(protocol_version, tuple):
        raise TypeError("protocol_version must be a tuple")
    
    result = {}
    for version, handler_class in protocol_handlers_map.items():
        if protocol_version is None or version >= protocol_version:
            result[version] = handler_class
    return result