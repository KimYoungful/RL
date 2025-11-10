from src.utils.logging import get_logger

def test_logger():
    logger = get_logger("test")
    logger.info("hello")
    assert logger is not None


