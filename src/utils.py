import logging
import sys


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(levelname)s:[%(asctime)s][%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    return logger
