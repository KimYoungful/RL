import logging



LOG_LEVEL = logging.INFO
SUBPROCESS_LOG_LEVEL = logging.ERROR
LOG_FORMATTER = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'


def get_logger(level=LOG_LEVEL, log_file=None, file_mode='w'):
    logger = logging.getLogger()
    logger.setLevel(level)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode=file_mode)
        file_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
    logger.addHandler(console_handler)
    return logger
