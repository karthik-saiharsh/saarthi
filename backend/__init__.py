"""Backend component: the control flow of the application."""

import logging

# One handler for all loggers of this package (`backend.ingest`, `backend.http`, ...).
_logger: logging.Logger = logging.getLogger(__name__)
if not _logger.handlers:
    _handler: logging.StreamHandler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s-%(asctime)s: %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
