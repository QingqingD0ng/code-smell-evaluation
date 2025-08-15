import logging

class YourClass:
    @classmethod
    def _reset_logging(cls):
        logging.shutdown()

# Example usage:
YourClass._reset_logging()