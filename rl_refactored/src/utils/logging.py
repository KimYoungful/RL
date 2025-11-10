import logging

LOG_LEVEL = logging.INFO
LOG_FORMATTER = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'

def get_logger(name: str = "rl", level: int = LOG_LEVEL) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(LOG_FORMATTER))
        logger.addHandler(console)
    return logger


