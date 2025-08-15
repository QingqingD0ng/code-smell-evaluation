import logging
import os
import logging.handlers

def build_app_logger(name='app', logfile='app.log', debug=True):
    log_directory = os.path.dirname(logfile)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    
    file_handler = logging.handlers.RotatingFileHandler(logfile, maxBytes=1048576, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger