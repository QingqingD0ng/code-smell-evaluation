def protocol_handlers(cls, protocol_version=None):

    if not isinstance(protocol_version, tuple) and protocol_version is not None:

        raise TypeError("protocol_version must be a tuple")


    handlers = {}

    for version, handler_cls in cls.PROTOCOL_VERSION_HANDLERS.items():

        if protocol_version is None or version == protocol_version:

            handlers[version] = handler_cls

    return handlers