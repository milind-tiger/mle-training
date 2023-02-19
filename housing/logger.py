import logging

from config import LOGS_PATH


def get_logger(logger_name):
    """
    get_logger : Generate logger object which will log the events in the module during the runtime

    Arguments:
        logger_name {str} -- Name of the process

    Returns:
        logger object -- Logger object to be used to log the events
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # open the text file used for logging in the append mode
    f_handler = logging.FileHandler(filename=LOGS_PATH, mode="w")
    # string format of the text to put in the log file
    f_format = logging.Formatter(
        "%(asctime)s - %(filename)s - %(funcName)s - %(lineno)d - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # assign the string format and file handler to the logger object
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    logger.addHandler(logging.StreamHandler())

    return logger
