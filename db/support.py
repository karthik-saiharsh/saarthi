"""Helpers that all parts of the db component share."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from peewee import IntegrityError, PeeweeException

from db.errors import DbError, DuplicateImageError, InvalidRecordError

logger: logging.Logger = logging.getLogger(__name__)


@contextmanager
def translate_errors(
    action: str, duplicate: type[DbError] = DuplicateImageError
) -> Iterator[None]:
    """Log peewee errors and raise them as the typed errors of this component.

    A unique constraint becomes `duplicate`. Another refused value becomes
    `InvalidRecordError`. Any other database error becomes `DbError`.
    """
    try:
        yield
    except IntegrityError as error:
        logger.error("%s failed: %s: %s", action, type(error).__name__, error)
        if "UNIQUE" in str(error).upper():
            raise duplicate(f"{action}: value already exists") from error
        raise InvalidRecordError(f"{action}: database refused the value") from error
    except PeeweeException as error:
        logger.error("%s failed: %s: %s", action, type(error).__name__, error)
        raise DbError(f"{action}: database error") from error
