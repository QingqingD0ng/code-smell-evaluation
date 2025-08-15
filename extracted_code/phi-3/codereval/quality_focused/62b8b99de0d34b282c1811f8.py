import logging

class LoggerManager:
    @classmethod
    def _reset_logging(cls):
        logging.getLogger().handlers = []
        logging.getLogger().setLevel(logging.NOTSET)
        logging.getLogger().addHandler(logging.NullHandler())