import logging
import logging.config
from os import environ

LOGGING_LEVEL_VAR = 'OH_LOGGING_LEVEL'

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
file_handlers = dict()
stream_handler = logging.StreamHandler()

def get_formatter():
    return formatter

def get_file_handler(file_name):
    if not file_name in file_handlers:
        file_handlers[file_name] = logging.FileHandler(file_name)
    
    return file_handlers[file_name]

def get_stream_handler():
    return stream_handler

def config_logger(logger: logging.Logger, file_name: str =None, level='INFO'):
    """
    Attach the correct handler and formatter to a given logger
    """
    if logger.hasHandlers():
        return
    
    if file_name:
        # Test write to file
        with open(file_name, 'w') as f:
            f.write('')

        handler = get_file_handler(file_name)
    else:
        handler = get_stream_handler()

    handler.setFormatter(get_formatter())

    # The environment variable overrides the level defined in program
    level = logging.getLevelName(environ.get(LOGGING_LEVEL_VAR, level))

    # Verify that it's a real logging level
    if type(level) == int:
        handler.setLevel(level)
        logger.setLevel(level)

    logger.addHandler(handler)

def getLogger(name=None, file_name: str =None, level='INFO'):
    logger = logging.getLogger(name)
    config_logger(logger, file_name, level)
    
    return logger
