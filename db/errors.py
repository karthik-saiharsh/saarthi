"""Errors of the db component."""


class ConfigError(Exception):
    """Raised when `config.toml` is missing or has a wrong value."""


class DbError(Exception):
    """Base class of the errors that the query helpers raise."""


class DuplicateImageError(DbError):
    """Raised when `file_path` or `sha256` of an image already exists."""


class InvalidStateError(DbError):
    """Raised when an image does not exist or has the wrong status."""


class InvalidRecordError(DbError):
    """Raised when the database refuses a value (for example a bad box)."""


class VectorIndexError(DbError):
    """Raised when the vector index cannot do what the caller asks."""


class DuplicateObjectError(DbError):
    """Raised when an enrolled object with this name already exists."""
