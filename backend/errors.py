"""Errors of the backend component."""


class BackendError(Exception):
    """Base class of the errors that the backend component raises."""


class BackendConfigError(BackendError):
    """Raised when `backend/config.toml` or the environment has a wrong value."""


class QueueFullError(BackendError):
    """Raised when too many images wait for the worker, so a new upload is refused."""
