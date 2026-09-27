"""Errors of the ai component."""


class AiError(Exception):
    """Base class of the errors that the ai component raises."""


class AiConfigError(AiError):
    """Raised when `ai/config.toml` is missing or has a wrong value."""


class ModelLoadError(AiError):
    """Raised when a model cannot be loaded."""


class ImageError(AiError):
    """Raised when an image cannot be used (bad file or too large)."""


class InferenceError(AiError):
    """Raised when a model fails while it works on an image."""


class CalibrationError(AiError):
    """Raised when the views are not enough for a calibration."""


class MatchingError(AiError):
    """Raised when the vectors or the objects of a matching call are not usable."""
