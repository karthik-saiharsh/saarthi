"""Errors of the backend component."""


class BackendError(Exception):
    """Base class of the errors that the backend component raises."""


class BackendConfigError(BackendError):
    """Raised when `backend/config.toml` or the environment has a wrong value."""


class QueueFullError(BackendError):
    """Raised when too many images wait for the worker, so a new upload is refused."""


class InvalidInputError(BackendError):
    """Raised when a value from the carer is not valid (a name or a phrase)."""


class LimitReachedError(BackendError):
    """Raised when the gallery has the largest number of objects or photos."""


class NotFoundError(BackendError):
    """Raised when an object or a photo does not exist."""


class ObjectNotReadyError(BackendError):
    """Raised when a search asks for an object that is a draft, or a look-alike that is never reported."""


class InvalidQueryError(BackendError):
    """Raised when the words of a user are empty, too long or not printable."""
