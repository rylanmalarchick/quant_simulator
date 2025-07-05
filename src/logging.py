import logging
import sys

def get_logger(name):
    """
    Returns a configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create a handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(handler)
    
    return logger
