
from kubeflow.common import structured_logging


def test_configure_logging_and_get_logger(monkeypatch):
    """Test that configure_logging succeeds and get_logger returns a structlog logger."""
    monkeypatch.setattr(structured_logging, "_CONFIGURED", False)

    structured_logging.configure_logging()

    logger = structured_logging.get_logger(__name__)

    assert logger is not None

    logger.info("test message")
