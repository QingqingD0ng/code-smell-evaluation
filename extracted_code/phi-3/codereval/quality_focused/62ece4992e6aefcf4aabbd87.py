import logging
from logging.handlers import RotatingFileHandler

def build_app_logger(name='app', logfile='app.log', debug=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if debug:
        handler = RotatingFileHandler(logfile, maxBytes=10000, backupCount=5)
    else:
        handler = logging.StreamHandler()
    
    handler.setFormatter(log_format)
    logger.addHandler(handler)
    
    return logger