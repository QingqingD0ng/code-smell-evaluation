import logging

class LoggerSystem:
    _reset_logging = classmethod(lambda cls:...)

    @classmethod
    def reset_logging(cls):
        logging.basicConfig(level=logging.NOTSET, handlers=[])
        logging.getLogger().handlers = []
        logging.getLogger().setLevel(logging.NOTSET)