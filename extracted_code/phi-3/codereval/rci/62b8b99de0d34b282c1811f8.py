import logging

class Application:
    @classmethod
    def _reset_logging(cls):
        if logging.root.handlers:
            for handler in logging.root.handlers:
                handler.close()
            logging.root.handlers.clear()
            logging.shutdown()