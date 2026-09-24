"""Errors of the storage component."""


class StorageError(Exception):
    """Base class of the errors that the storage component raises."""


class StorageConfigError(StorageError):
    """Raised when `storage/config.toml` is missing or has a wrong value."""


class UnsupportedImageError(StorageError):
    """Raised when data is not a usable image (format, animation, broken or empty file)."""


class ImageTooLargeError(StorageError):
    """Raised when an image has too many bytes or too many pixels."""


class StorageFullError(StorageError):
    """Raised when the disk has too little free space for a new image."""


class StoragePathError(StorageError):
    """Raised when a path is not a safe path inside the storage root."""


class StorageNotFoundError(StorageError):
    """Raised when a stored file does not exist."""


class StorageIOError(StorageError):
    """Raised when the disk refuses a read, a write or a delete."""


class InvalidCropError(StorageError):
    """Raised when a crop box does not lie inside the image."""
